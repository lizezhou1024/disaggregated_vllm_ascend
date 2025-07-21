# SPDX-License-Identifier: Apache-2.0
"""
    This module implements a PyNccl pipe for sending and receiving 
    Optional[torch.Tensor] between distributed ranks with advanced 
    communication features.

    Key Features:
    - Supports sending and receiving tensors with metadata
    - Handles both CUDA and CPU device communications
    - Implements a non-blocking tensor transfer mechanism
    - Manages buffer size and provides backpressure control
    - Supports distributed process groups with configurable parameters
"""

import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple, List, Union
import torch.distributed as dist
from torch.distributed import ProcessGroup
import pickle
import torch
import traceback
from functools import partial

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

from vllm.utils import (get_distributed_init_method,
                        get_open_port)

logger = init_logger(__name__)


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


Metadata = Dict[str, Optional[torch.Tensor]]


class PyNcclPipe(KVPipeBase):

    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None,
                 port_offset: int = 0,
                 group: Optional[List[ProcessGroup]] = None,
                 peer_ranks: Optional[List[int]] = None,
                 union_scale: Optional[int] = None):
        self.config = config
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        self.is_procuder = self.config.is_kv_producer
        self.kv_parallel_size = self.config.kv_parallel_size
        self.role = config.kv_role
        if device is None:
            self.device = self._select_device(self.config.kv_buffer_device)
            self.backend = "hccl"
        else:
            self.device = self._select_device(device)
            self.backend = "gloo"

        self.peer_ranks = peer_ranks
        self.num_peer_kv_ranks = len(peer_ranks)
        self.num_peers = len(peer_ranks[0])

        assert len(group) == self.num_peer_kv_ranks        # Use one group for each peer kv rank
        self.group = group
        assert self.group is not None

        self.my_rank = dist.get_rank()
        self.rank_in_group = dist.get_rank(self.group[0])

        logger.info("is_procuder: %d, kv_rank: %d, kv_parallel_size: %d", self.is_procuder, self.kv_rank,
                    self.kv_parallel_size)
        logger.info(f"peer_ranks: {self.peer_ranks}")
        # transportation-related variables
        self.transport_thread: List[Optional[ThreadPoolExecutor]] = [None for _ in range(self.num_peer_kv_ranks)]
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()
        self.buffer_size_thresh = self.config.kv_buffer_size

        self.union_scale = union_scale

    def _select_device(self, device: str):
        logger.info("Selecting device: %s ,local_rank %s", device, self.local_rank)
        if device == "cuda":
            return torch.device(f"cuda:{self.local_rank}")  
        elif device == "npu":
            return torch.device(f"npu:{self.local_rank}")
        else:
            return torch.device("cpu")

    def _make_metadata(self, tensor: Optional[torch.Tensor]) -> Metadata:
        """
        Create the metadata as a dictionary based on the input tensor.

        Parameters:
            - tensor: The input tensor or None if no tensor is provided.

        Returns:
            - metadata: A dictionary with the following keys:
                - "dtype": The data type of the tensor or None.
                - "shape": The shape of the tensor or None.
        """
        if tensor is None:
            return {"dtype": None, "shape": None}
        else:
            return {"dtype": tensor.dtype, "shape": tensor.shape}

    def _prepare_recv_buffer(self, metadata: Metadata) -> torch.Tensor:
        """
        Create a buffer to receive the tensor based on the provided metadata.

        Parameters:
            - metadata: A dictionary with keys "dtype" and "shape", describing 
              the tensor's data type and shape.

        Returns:
            - buffer: A tensor of the specified type and shape, allocated on 
              self.device.
        """
        return torch.empty(metadata["shape"],
                           dtype=metadata["dtype"],
                           device=self.device)

    def _send_metadata(self, metadata: Metadata, peer_kv_rank: int, peer_id: int, use_broadcast: bool = False):
        """
        Send the metadata dictionary to the target rank.

        Parameters:
            - metadata: A dictionary with keys "dtype" and "shape".
        """
        # print(f"in _send_metadata: peer = {peer_id}, use_broadcast = {use_broadcast}")
        metadata_bytes = pickle.dumps(metadata)
        metadata_size = torch.tensor([len(metadata_bytes)], dtype=torch.long, device=self.device)
        metadata_tensor = torch.tensor(bytearray(metadata_bytes), dtype=torch.uint8, device=self.device)
        if use_broadcast:
            dist.broadcast(metadata_size, src=self.my_rank, group=self.group[peer_kv_rank])
            dist.broadcast(metadata_tensor, src=self.my_rank, group=self.group[peer_kv_rank])
        else:
            dist.send(metadata_size, self.peer_ranks[peer_kv_rank][peer_id], group=self.group[peer_kv_rank])
            dist.send(metadata_tensor, self.peer_ranks[peer_kv_rank][peer_id], group=self.group[peer_kv_rank])
        # print(f"succ _send_metadata: peer = {peer_id}, use_broadcast = {use_broadcast}")

    def _recv_metadata(self, peer_kv_rank: int, peer_id: List[int], use_broadcast: bool = False) -> Metadata:
        """
        Receive the metadata dictionary from the target rank.

        Returns:
            - metadata: A dictionary with keys "dtype" and "shape" describing 
              the tensor.
        """
        # print(f"in _recv_metadata: peer = {peer_id}, use_broadcast = {use_broadcast}")
        metadata_size = torch.zeros(1, dtype=torch.long, device=self.device)
        if use_broadcast:
            dist.broadcast(metadata_size, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])
        else:
            dist.recv(metadata_size, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])
        
        metadata_tensor = torch.zeros(metadata_size.item(), dtype=torch.uint8, device=self.device)
        if use_broadcast:
            dist.broadcast(metadata_tensor, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])
        else:
            dist.recv(metadata_tensor, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])
        metadata_bytes = metadata_tensor.cpu().numpy().tobytes()
        metadata = pickle.loads(metadata_bytes)
        # print(f"succ _recv_metadata: peer = {peer_id}, use_broadcast = {use_broadcast}")
        return metadata

    def _send_impl(self, tensor_or_list: Optional[Union[torch.Tensor, List[torch.Tensor]]], peer_kv_rank: int) -> None:
        """
        The actual implementation of sending the tensor and its metadata to the 
        target rank.

        Parameters:
            - tensor: The input tensor to be sent, or None if no tensor is 
              being sent.
        """
        # print("[rank%d]: Sending tensor to rank %d", self.kv_rank, self.target_rank_for_send)
        use_broadcast = False       # single tensor send to multiple dests
        use_scatter = False         # multiple tensor send to multiple dests
        use_gather = False          # multiple tensor send to single dest
        peer_ids = list(range(self.num_peers))
        if isinstance(tensor_or_list, list):
            assert(len(tensor_or_list) == self.num_peers)
            tensors = tensor_or_list
        else:
            tensors = [tensor_or_list for _ in range(self.num_peers)]
        if self.num_peers == 1 and self.union_scale > 1 and self.is_procuder:
            use_gather = True
        elif self.num_peers > 1:
            if self.is_procuder:
                # we are kv sender
                use_scatter = True
            else:
                # we are kv receiver
                use_broadcast = True

        # print(f"start to send tensor, device = {self.device}, peer_id = {peer_ids}, use_broadcast = {use_broadcast}, use_scatter = {use_scatter}")

        if use_broadcast:
            metadata = self._make_metadata(tensor_or_list)
            self._send_metadata(metadata, peer_kv_rank, peer_ids, True)
            dist.broadcast(tensor_or_list.to(self.device), src=self.my_rank, group=self.group[peer_kv_rank])
        elif use_gather:
            if self.rank_in_group == 0:
                metadata = self._make_metadata(tensors[0])
                self._send_metadata(metadata, peer_kv_rank, peer_ids[0])
            
            dist.send(tensors[0].to(self.device), self.peer_ranks[peer_kv_rank][0], group=self.group[peer_kv_rank])
        elif use_scatter:
            metadata = self._make_metadata(tensors[0])
            # use broadcast to send metadata
            self._send_metadata(metadata, peer_kv_rank, peer_ids, True)
            for i in range(self.num_peers):
                tensors[i] = tensors[i].to(self.device)
            temp_tensor = tensors[0].clone()
            tensors.insert(self.rank_in_group, temp_tensor)
            dist.scatter(scatter_list=tensors, tensor=temp_tensor, src=self.my_rank, group=self.group[peer_kv_rank])
        else:
            for index, peer_id in enumerate(peer_ids):
                # print(f"in _send_impl: peer = {peer_id}, rank = {self.peer_ranks[peer_id]}")
                tensor = tensors[index]
                metadata = self._make_metadata(tensor)
                self._send_metadata(metadata, peer_kv_rank, peer_id)
                if tensor is not None:
                    dist.send(tensor.to(self.device), self.peer_ranks[peer_kv_rank][peer_id], group=self.group[peer_kv_rank])
                # print(f"succ _send_impl: peer = {peer_id}, rank = {self.peer_ranks[peer_id]}")
        
        # print(f"end send tensor, device = {self.device}, peer_id = {peer_ids}, use_broadcast = {use_broadcast}, use_scatter = {use_scatter}")

    def _recv_impl(self, peer_kv_rank: int, peer_id: Optional[List[int]] = None) -> Optional[torch.Tensor]:
        """
        The actual implementation of receiving a tensor and its metadata from 
        the target rank.

        Returns:
            - buffer: The received tensor, or None if no tensor is received.
        """

        # print("[rank%d]: Receiving tensor from rank %d", self.kv_rank,self.target_rank_for_recv)  
        #print(f"Recving tensor...")
        if peer_id is None:
            peer_id = [0]
        if len(peer_id) > 1:
            multi_input = True
        else:
            multi_input = False

        use_broadcast = False
        use_scatter = False
        if self.union_scale > 1:
            if self.is_procuder:
                # We are kv sender. Use broadcast to receive roi.
                use_broadcast = True
            else:
                # We are kv receiver. Use scatter to receive kv.
                use_scatter = True
        
        # print(f"start to recv tensor, device = {self.device}, peer_id = {peer_id}, use_broadcast = {use_broadcast}, use_scatter = {use_scatter}")
        metadata = self._recv_metadata(peer_kv_rank, peer_id, use_broadcast or use_scatter)
        if metadata["dtype"] is None:
            return None
        
        if multi_input:
            buffer = [self._prepare_recv_buffer(metadata) for _ in peer_id]
        else:
            buffer = self._prepare_recv_buffer(metadata)

        if multi_input:
            for id in peer_id:
                dist.recv(buffer[id], src=self.peer_ranks[peer_kv_rank][id], group=self.group[peer_kv_rank])
        elif use_broadcast:
            dist.broadcast(buffer, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])
        elif use_scatter:
            dist.scatter(scatter_list=None, tensor=buffer, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])
        else:
            dist.recv(buffer, src=self.peer_ranks[peer_kv_rank][peer_id[0]], group=self.group[peer_kv_rank])

        # print(f"end recv tensor {metadata['shape']}, device = {self.device}, peer_id = {peer_id}, use_broadcast = {use_broadcast}, use_scatter = {use_scatter}")

        if isinstance(buffer, list):
            for i in range(len(buffer)):
                if not buffer[i].is_npu:
                    buffer[i] = buffer[i].to("npu:" + str(self.local_rank))
        elif isinstance(buffer, torch.Tensor):
            if not buffer.is_npu:
                buffer = buffer.to("npu:" + str(self.local_rank))

        return buffer

    def send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int, peer_kv_rank: int) -> None:
        """
        Wrapper for _send_impl to handle exceptions and update buffer size.
        """
        try:
            # print(f"start to send {tensor_size}, device = {self.device}")
            self._send_impl(tensor, peer_kv_rank)
            # print(f"end send {tensor_size}, device = {self.device}")

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):
        """
        Block the current thread if the buffer size is larger than the 
        threshold.
        """
        while self.buffer_size > self.buffer_size_thresh:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self, tensor_or_list: Optional[torch.Tensor], peer_kv_rank: int) -> None:
        """
        Sends a tensor and its metadata to the destination rank in a 
        non-blocking way.

        Parameters:
            - tensor: The tensor to send, or None if no tensor is being sent.
        """
        if self.transport_thread[peer_kv_rank] is None:
            self.transport_thread[peer_kv_rank] = ThreadPoolExecutor(max_workers=1)

        if isinstance(tensor_or_list, torch.Tensor):
            tensor_size = tensor_or_list.element_size() * tensor_or_list.numel()
        elif isinstance(tensor_or_list, List) and isinstance(tensor_or_list[0], torch.Tensor):
            tensor_size = 0
            for tensor in tensor_or_list:
                tensor_size += tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size += tensor_size

        self.transport_thread[peer_kv_rank].submit(self.send_tensor_wrapper, tensor_or_list,
                                     tensor_size, peer_kv_rank)

    def recv_tensor(self, peer_kv_rank: int, peer_id: Optional[List[int]] = None) -> Optional[torch.Tensor]:
        """
        Receives a tensor and its metadata from the source rank. Blocking call.

        Returns:
            - tensor: The received tensor, or None if no tensor is received.
        """
        if self.transport_thread[peer_kv_rank] is None:
            self.transport_thread[peer_kv_rank] = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread[peer_kv_rank].submit(self._recv_impl, peer_kv_rank, peer_id)

        try:
            tensor = future.result()
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            dist.destroy_process_group()
            logger.error("My device: %s", self.device)
            import traceback
            traceback.print_exc()
            raise e

        return tensor

    def close(self):
        """
        Close the pipe and release associated resources.
        """
        dist.destroy_process_group()

        if hasattr(self, "transport_thread"):
            for thread in self.transport_thread:
                if thread is not None:
                    thread.shutdown()
