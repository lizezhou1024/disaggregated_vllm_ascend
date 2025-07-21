# SPDX-License-Identifier: Apache-2.0
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to 
      stop the prefill instance when the decode instance is slow.
"""
import threading
from collections import deque
from typing import Deque, List, Optional, Union

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float, num_peers: int, peer_kv_parallel_size: int):
        """
        signal_pipe: on CPU

        NOTE: on-device recv will block all threads in the process, making the
        KV cache producer unable to listen to new request while transmitting
        KV cache. Luckily CPU recv only blocks the current thread so we use
        CPU recv to listen to new request.

        data_pipe: on device (e.g. GPU)
        """

        self.buffer: List[Deque[List[torch.Tensor]]] = [deque() for _ in range(peer_kv_parallel_size)]

        self.buffer_size = [0 for _ in range(peer_kv_parallel_size)]
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_cv = [threading.Condition() for _ in range(peer_kv_parallel_size)]
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: List[Optional[threading.Thread]] = [None for _ in range(peer_kv_parallel_size)]
        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None
        self.num_peers = num_peers

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor], peer_kv_rank: int) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size[peer_kv_rank] -= self._get_element_size(tensor)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipe.send_tensor(tensor, peer_kv_rank)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if isinstance(data, list) and isinstance(data[0], torch.Tensor):
            sum = 0
            for tensor in data:
                sum += tensor.element_size() * tensor.numel()
            return sum
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor, peer_kv_rank: int):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key[0], torch.Tensor):
            key = [tensor.clone() for tensor in key]
        if isinstance(value[0], torch.Tensor):
            value = [tensor.clone() for tensor in value]
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, roi, key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv[peer_kv_rank]:
            if self.buffer_size[peer_kv_rank] + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                print(f"connector: KV transfer buffer is full. Handling...", flush=True)
                while self.buffer_size[peer_kv_rank] + data_size > self.buffer_size_threshold:
                    self.buffer_cv[peer_kv_rank].wait()

            self.buffer_size[peer_kv_rank] += data_size
            self.buffer[peer_kv_rank].append(buffer_item)
            self.buffer_cv[peer_kv_rank].notify()

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self, peer_kv_rank: int):

        try:

            while True:
                signal = self.signal_pipe.recv_tensor(peer_kv_rank)
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor(peer_kv_rank)

                roi = self.data_pipe.recv_tensor(peer_kv_rank)
                assert roi is not None, "Please provide the roi when sending "\
                    "drop-select request"
                roi = (roi > 0.5)
                tokens_roi_recver = [input_tokens, roi]

                def is_buffer_available(
                    tokens_roi_recver: List[torch.Tensor], ) -> bool:
                    # perform input tokens and roi matching
                    # FIXME: this matching is O(n), ideally it should be O(1)
                    # but this buffer size won't (and shouldn't) be too large so
                    # the fix is not urgent.
                    buffer = self.buffer[peer_kv_rank]
                    for _ in range(len(buffer)):
                        if self._matches(buffer[0],
                                         tokens_roi_recver) > 0:
                            return True
                        # rotate the element we just accessed to the end
                        buffer.rotate(-1)
                    return False

                with self.buffer_cv[peer_kv_rank]:
                    while not is_buffer_available(tokens_roi_recver):
                        logger.debug(
                            "KV transfer buffer is not available. Waiting...")
                        self.buffer_cv[peer_kv_rank].wait()
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer[peer_kv_rank].popleft()
                    for tensor in matched_item:
                        self._send_tensor_and_dec_size(tensor, peer_kv_rank)
                    self.buffer_cv[peer_kv_rank].notify()

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor],
            peer_kv_rank: int,
            is_master_in_group: bool = True) -> List[Optional[torch.Tensor]]:

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone().float()

        # Only need to send info for master
        peer_ids = list(range(self.num_peers))
        if is_master_in_group:
            self.signal_pipe.send_tensor(self.normal_signal, peer_kv_rank)
            self.data_pipe.send_tensor(input_tokens, peer_kv_rank)
            self.data_pipe.send_tensor(roi, peer_kv_rank)

        input_tokens = self.data_pipe.recv_tensor(peer_kv_rank, peer_ids)
        rois = self.data_pipe.recv_tensor(peer_kv_rank, peer_ids)
        keys = self.data_pipe.recv_tensor(peer_kv_rank, peer_ids)
        values = self.data_pipe.recv_tensor(peer_kv_rank, peer_ids)
        hiddens = self.data_pipe.recv_tensor(peer_kv_rank, peer_ids)

        if len(peer_ids) > 1:
            rois_ = []
            for roi in rois:
                if roi is not None:
                    # convert from float tensor to bool tensor
                    # as PyNccl does not support sending bool tensor
                    roi = (roi > 0.5)
                rois_.append(roi)
            ret = []
            for i in range(self.num_peers):
                ret.append([input_tokens[i], rois_[i], keys[i], values[i], hiddens[i]])
        else:
            if rois is not None:
                # convert from float tensor to bool tensor
                # as PyNccl does not support sending bool tensor
                rois = (rois > 0.5)
            ret = [[input_tokens, rois, keys, values, hiddens]]
        
        return ret

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor, peer_kv_rank: int) -> None:

        self._add_to_buffer(input_tokens, roi, key, value, hidden, peer_kv_rank)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread[peer_kv_rank] is None:
            self.request_handling_thread[peer_kv_rank] = threading.Thread(
                target=self.drop_select_handler, args=(peer_kv_rank,))
            self.request_handling_thread[peer_kv_rank].start()

    def close(self):

        if hasattr(self, "request_handling_thread"):
            for thread in self.request_handling_thread:
                if thread is not None:
                    thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
