# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import time
import torch
import torch_npu

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class SimpleConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
        group: Tuple[List["DeviceGroup"], List["CPUGroup"]],
    ):

        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size

        if self.config.kv_connector == "PyNcclConnector":
            from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
                PyNcclPipe)
            # self.config.kv_buffer_device = 'npu' 
            self.config.kv_buffer_device = 'cpu' 
            logger.info(
                "Initializing PyNcclConfig under kv_transfer_config %s",
                self.config)
        elif self.config.kv_connector == "MooncakeConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_distributed_pipe = os.getenv(
                'MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_distributed_pipe:
                raise ValueError(
                    "To use MooncakeConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import (  # noqa: E501
                    MooncakePipe)
                logger.info(
                    "Initializing MooncakeConfig under kv_transfer_config %s",
                    self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None

        self.producer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.producer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_signal_pipe: Union[PyNcclPipe, MooncakePipe]

        self.kv_rank = self.config.kv_rank

        self.should_split = False
        self.should_union = False
        self.split_scale = 1
        self.union_scale = 1
        if self.config.peer_world_size > config.parallel_config.world_size:
            self.should_split = True
            self.split_scale = self.config.peer_world_size // config.parallel_config.world_size
        elif self.config.peer_world_size < config.parallel_config.world_size:
            self.should_union = True
            self.union_scale = config.parallel_config.world_size // self.config.peer_world_size
        
        self.is_master_in_group = True
        rank_in_group = rank - config.parallel_config.world_size * self.kv_rank - (self.config.peer_kv_parallel_size * self.config.peer_world_size if self.config.is_kv_consumer else 0)
        if self.should_union and rank_in_group % self.union_scale > 0:
            self.is_master_in_group = False
        
        print(f"connector rank {rank}, rank_in_group {rank_in_group}, split_scale = {self.split_scale}, union_scale = {self.union_scale}")

        # 2 pipes for every rank in the world
        port_offset_base = 2 * rank

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:

            if self.config.kv_connector == "PyNcclConnector":
                peer_ranks = []
                for peer_kv_rank in range(self.config.peer_kv_parallel_size):
                    base_rank = self.config.kv_parallel_size * config.parallel_config.world_size + peer_kv_rank * self.config.peer_world_size
                    ranks = []
                    for i in range(self.split_scale):
                        if self.should_split:
                            peer_rank = base_rank + rank_in_group * self.split_scale+i
                        elif self.should_union:
                            peer_rank = base_rank + (rank_in_group // self.union_scale)
                        else:
                            peer_rank = base_rank + rank_in_group
                        ranks.append(peer_rank)
                    peer_ranks.append(ranks)

                self.producer_data_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                    group=group[0],
                    peer_ranks=peer_ranks,
                    union_scale=self.union_scale,
                )
                self.producer_signal_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                    group=group[1],
                    peer_ranks=peer_ranks,
                    union_scale=self.union_scale,
                )
            elif self.config.kv_connector == "MooncakeConnector":
                self.producer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                # We only need to initialize MooncakePipe once
                self.producer_signal_pipe = self.producer_data_pipe

            self.producer_buffer = SimpleBuffer(self.producer_signal_pipe,
                                                self.producer_data_pipe,
                                                self.config.kv_buffer_size,
                                                self.split_scale,
                                                self.config.peer_kv_parallel_size)

        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder
            if self.config.kv_connector == "PyNcclConnector":
                peer_ranks = []
                for peer_kv_rank in range(self.config.peer_kv_parallel_size):
                    base_rank = peer_kv_rank * self.config.peer_world_size
                    ranks = []
                    for i in range(self.split_scale):
                        if self.should_split:
                            peer_rank = base_rank + rank_in_group * self.split_scale+i
                        elif self.should_union:
                            peer_rank = base_rank + (rank_in_group // self.union_scale)
                        else:
                            peer_rank = base_rank + rank_in_group
                        ranks.append(peer_rank)
                    peer_ranks.append(ranks)
                        
                self.consumer_data_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                    group=group[0],
                    peer_ranks=peer_ranks,
                    union_scale=self.union_scale,
                )
                self.consumer_signal_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                    group=group[1],
                    peer_ranks=peer_ranks,
                    union_scale=self.union_scale,
                )
            elif self.config.kv_connector == "MooncakeConnector":
                self.consumer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                self.consumer_signal_pipe = self.consumer_data_pipe

            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
                self.split_scale,
                self.config.peer_kv_parallel_size,
            )

    def select(self, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor], peer_kv_rank) -> List[Optional[torch.Tensor]]:

        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(input_tokens, roi, peer_kv_rank, self.is_master_in_group)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor, peer_kv_rank: int) -> None:

        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(input_tokens, roi, key, value, hidden, peer_kv_rank)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        # print(f"connector: start send_kv_caches_and_hidden_states")
        # func_stime = time.time()
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        # print(f"connector: process {len(seq_lens)} seqs", flush=True)
        for idx, slen in enumerate(seq_lens):
            # stime = time.time()
            peer_kv_rank = model_input.peer_kv_ranks[idx]
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [[] for _ in range(self.split_scale)], [[] for _ in range(self.split_scale)]

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, self.split_scale, num_heads // self.split_scale, head_size)
                value_cache = kv_cache[1].reshape(-1, self.split_scale, num_heads // self.split_scale, head_size)


                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys_split = torch.unbind(key_cache[current_slot_mapping], dim=1)
                values_split = torch.unbind(value_cache[current_slot_mapping], dim=1)

                for split_id in range(self.split_scale):
                    keys[split_id].append(keys_split[split_id].unsqueeze(0))
                    values[split_id].append(values_split[split_id].unsqueeze(0))

            for split_id in range(self.split_scale):
                keys[split_id] = torch.cat(keys[split_id], dim=0).contiguous()
                values[split_id] = torch.cat(values[split_id], dim=0).contiguous()
            # print(f"connector: start to insert {idx}, slen = {slen}, shape = {keys[0].shape}, start_pos = {start_pos}, end_pos = {end_pos}", flush=True)

            # print(f"send kv: scale = {self.split_scale}, roi = {current_tokens.shape}, key = {keys[0].shape}, value = {values[0].shape}, hidden = {hidden_or_intermediate_states[start_pos:end_pos].shape}")

            # etime = time.time()
            # print(f"connector: {idx} split time = {etime - stime}", flush=True)
            # stime = etime

            self.insert(current_tokens,
                        torch.ones_like(current_tokens,
                                        dtype=bool), keys, values,
                        hidden_or_intermediate_states[start_pos:end_pos], peer_kv_rank)
            # etime = time.time()
            # print(f"connector: {idx} insert time = {etime - stime}", flush=True)
        # print(f"connector: end send_kv_caches_and_hidden_states, time cost = {time.time() - func_stime}")

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        # print(f"connector: start recv_kv_caches_and_hidden_states")
        # func_stime = time.time()

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            peer_kv_rank = model_input.peer_kv_ranks[idx]

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self.select(current_tokens,
                              torch.ones_like(current_tokens, dtype=bool), peer_kv_rank)
            if ret[0][0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[0][1]
            keys: torch.Tensor = [item[2] for item in ret]
            values: torch.Tensor = [item[3] for item in ret]
            hidden: torch.Tensor = ret[0][4]

            # print(f"recv kv: roi = {roi.shape}, keys = {keys[0].shape}, values = {values[0].shape}, hidden = {hidden.shape}, kv_cache_layer = {kv_caches[0][0].shape}")

            # Concate on the kv_head dimension
            keys = torch.cat(keys, dim=2)
            values = torch.cat(values, dim=2)

            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            """keys = keys.to(kv_caches[0].device)
            values = values.to(kv_caches[0].device)
            slot_mapping_ = slot_mapping[start_pos:end_pos].to("cpu")

            # Optimization: merge mapping into intervals
            num_tokens = keys.size(1)
            mapping_intervals = []
            i = 0
            while i < num_tokens:
                idx = slot_mapping_[i]
                j = i + 1
                while j < num_tokens:
                    if slot_mapping_[j] != idx+j-i:
                        break
                    j += 1
                mapping_intervals.append((i,j))
                i = j"""

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                key_cache, value_cache = kv_cache[0], kv_cache[1]
                """
                ops.reshape_and_cache_flash(
                    keys[i - model_executable.model.start_layer].to(
                        key_cache.device),
                    values[i - model_executable.model.start_layer].to(
                        value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )
                """
                torch_npu._npu_reshape_and_cache(
                    key=keys[i - model_executable.model.start_layer].to(
                        key_cache.device),
                    value=values[i - model_executable.model.start_layer].to(
                        value_cache.device),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_indices=slot_mapping[start_pos:end_pos],
                    #layer.self_attn.attn.kv_cache_dtype,
                    #layer.self_attn.attn._k_scale,
                    #layer.self_attn.attn._v_scale,
                )
                
                """# Simulate reshape_and_cache as Ascend does not have this operator
                key_cache = key_cache.view(-1, key_cache.size(2), key_cache.size(3))
                value_cache = value_cache.view(-1, value_cache.size(2), value_cache.size(3))
                key = keys[i - model_executable.model.start_layer]
                value = values[i - model_executable.model.start_layer]
                for s, t in mapping_intervals:
                    idx = slot_mapping_[s]
                    key_cache[idx:idx+t-s] = key[s:t]
                    value_cache[idx:idx+t-s] = value[s:t]"""

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        # print(f"connector: end recv_kv_caches_and_hidden_states, time cost = {time.time() - func_stime}")
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        for split_id in range(self.split_scale):
            self.producer_data_pipe[split_id].close()
            self.consumer_data_pipe[split_id].close()
        if self.config.kv_connector == "PyNcclConnector":
            for split_id in range(self.split_scale):
                self.producer_signal_pipe[split_id].close()
                self.consumer_signal_pipe[split_id].close()
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass
