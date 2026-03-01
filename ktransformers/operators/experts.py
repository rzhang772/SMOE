#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-08-29 09:41:10
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''

from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from ktransformers.operators.base_operator import BaseInjectedModule
from tqdm import tqdm
import nvtx

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
# import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
import ctypes
from ktransformers.util.custom_gguf import GGMLQuantizationType
from ktransformers.util.custom_loader import GGUFLoader, SafeTensorLoader, ModelLoader
from ktransformers.util.utils import InferenceState
from ktransformers.server.config.config import Config
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from ktransformers.operators.linear import KLinearMarlin, KLinearTorch, KTransformersLinear, SLinear, SLinearWrap
import time
from ktransformers.operators.cpuinfer import CPUInfer
from ktransformers.operators.predictor_model import TopkPredictor


def deduplicate_and_sort(lst):
    return sorted(set(lst))
def generate_cuda_graphs(chunk_size: int) -> list:
    assert chunk_size <= 1024 or chunk_size % 1024 == 0, "chunk_size must <= 1024 or a multiple of 1024"
    base_list = [1, 2, 3, Config().max_batch_size, 64, 256, 512, chunk_size]

    if chunk_size <= 1024:
        return deduplicate_and_sort(base_list)

    multiples = [i for i in range(1024, chunk_size + 1, 1024)]

    return deduplicate_and_sort(base_list + multiples)
#cuda_graphs = [Config().chunk_size] 

# cuda_graphs 是一个列表
if torch.cuda.is_available():
    cuda_graphs = generate_cuda_graphs(Config().chunk_size)
else:
    cuda_graphs = 1
# class Base(BaseInjectedModule, ABC):
class KExpertsBase(ABC):
    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str = "cuda", **kwargs):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
    
    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu", warmup: bool = False):
        pass
    
    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                targets = [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" ]
                tensors = self.load_multi(key, targets, device=device)
                gate = tensors[".ffn_gate_exps.weight"]
                up = tensors[".ffn_up_exps.weight"]
                down = tensors[".ffn_down_exps.weight"]
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif self.gguf_loader.has_tensor(key + ".ffn_down.0.weight"):
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gatei, upi, downi = f".ffn_gate.{i}.weight", f".ffn_up.{i}.weight", f".ffn_down.{i}.weight"
                    targets = [gatei, upi, downi]
                    tensors = self.load_multi(key, targets, device=device)
                    gate_it, up_it, down_it = tensors[gatei], tensors[upi], tensors[downi]
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = torch.stack(gate)
                up = torch.stack(up)
                down = torch.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors


# Revised KExpertsCPU class for SMOE
class KExpertsCPU(KExpertsBase):
    input_tensor_cpu:Tensor = None
    expert_ids_cpu:Tensor = None
    weights_cpu:Tensor = None
    output_cpu:Tensor = None
    output_gpu_map:dict = {} # Manage output tensor buffer on different gpu

    prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    layer_counter = 0
    hit_rate = []

    # [0, 58]
    prefetch_layers = [i for i in range(Config().prefetch_start_layer, 58)]

    #stream_map:dict = {} # Manage cuda stream on different gpu
    # @TODO add yaml
    CPU_INFER = CPUInfer(Config().cpu_infer)
    print(f"----------------------------------------------------------------------------------------------------CPU_INFER in KExpertsCPU: {Config().cpu_infer}")

    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda", # this device mean which device the output should on. TODO: support cpu.
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        # print(f"=================>>>>. {self.key}: KExpertsCPU initialized, {[self.key]}")
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device # 也就是generate_device, 这个参数是为了支持在cpu上运行的模型，输出tensor会被放到out_device上, 这里是cuda
        self.backend = kwargs.get("backend", "llamafile") # 从参数中获取backend类型，默认为llamafile


        # SMOE: expert cache 初始化
        self.print_layer = 10
        self.gpu_device = "cuda"
        self.cpu_device = "cpu"
        self.cached_experts_num = 8
        # 使用shared_experts的key来初始化，因为这里shared和routed结构相同
        shared_key = key[: -len(".experts")] + ".shared_experts"
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.elements_per_expert = self.moe_intermediate_size * self.hidden_size
        target_dtype = torch.get_default_dtype()
        # self.cached_experts_wrap = SLinearWrap(self.moe_intermediate_size, self.hidden_size, gguf_loader, config, target_dtype=target_dtype, device=self.gpu_device)
        self.cached_experts = {
            # up
            "up_projs" : [SLinear(self.hidden_size, self.moe_intermediate_size, gguf_loader, config, target_dtype=target_dtype, linear_type = "up", device=self.gpu_device) for i in range(self.cached_experts_num)],
            # gate
            "gate_projs" : [SLinear(self.hidden_size, self.moe_intermediate_size, gguf_loader, config, target_dtype=target_dtype, linear_type = "gate", device=self.gpu_device) for i in range(self.cached_experts_num)],
            # down
            "down_projs" : [SLinear(self.moe_intermediate_size, self.hidden_size, gguf_loader, config, target_dtype=target_dtype, linear_type = "down", device=self.gpu_device) for i in range(self.cached_experts_num)]
        }
        self.up_ggml_size = self.gguf_loader.get_expert_ggml_size(key + ".ffn_up_exps.weight")
        self.gate_ggml_size = self.gguf_loader.get_expert_ggml_size(key + ".ffn_gate_exps.weight")
        self.down_ggml_size = self.gguf_loader.get_expert_ggml_size(key + ".ffn_down_exps.weight")
        # print(up_ggml_size, gate_ggml_size, down_ggml_size)
        # print(self.moe_intermediate_size * up_ggml_size, self.moe_intermediate_size * gate_ggml_size, self.hidden_size * down_ggml_size)

        self.up_slots   = None
        self.gate_slots = None
        self.down_slots = None
        self.up_slots_ptr   = None
        self.gate_slots_ptr = None
        self.down_slots_ptr = None

        self.cached_experts_ids = None # 用于缓存专家的id
        # self.prefetch_event = torch.cuda.Event(blocking=True)
        self.cache_ready = torch.zeros(1, dtype=torch.int32, device='cpu')
        # print(f"just initialized: {self.prefetch_event.query()}")
        self.layer_id = KExpertsCPU.layer_counter
        KExpertsCPU.layer_counter += 1
        self.expert_frequency = torch.zeros(256, dtype=torch.int64, device='cpu')
        

        self.predictor_path = f"/mnt/incontainer/shared_rui/predictors/top8_alldataset_singleLinear_B32/layer_{self.layer_id}/best_model_layer_{self.layer_id}.pth"
        self.predictor = TopkPredictor(input_dim=7168, expert_num=256).to_empty(device=self.gpu_device)
        self.predictor.load_state_dict(torch.load(self.predictor_path, map_location=self.gpu_device))
        self.predictor.eval()

        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_prefetch_ready = torch.zeros(1, dtype=torch.int32, device='cpu')


    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None, warmup:bool = False):
        if device:
            assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU, Parameter \"device\" can be cpu or None."
        if w is None: w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]

        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        # print(f"{w['gate'].shape}, {w['up'].shape}, {w['down'].shape}")
        # self.gate = torch.from_numpy(w["gate"]).pin_memory()
        # self.up   = torch.from_numpy(w["up"]).pin_memory()
        # self.down = torch.from_numpy(w["down"]).pin_memory()

        # self.gate_type = w["gate_type"]
        # self.up_type   = w["up_type"]
        # self.down_type = w["down_type"]

        # gate_ptr = self.gate.data_ptr()
        # up_ptr   = self.up.data_ptr()
        # down_ptr = self.down.data_ptr()
        
        self.up_slots   = [torch.empty(self.moe_intermediate_size * self.up_ggml_size, device=self.gpu_device, dtype=torch.uint8) for _ in range(self.cached_experts_num)]
        self.gate_slots = [torch.empty(self.moe_intermediate_size * self.gate_ggml_size, device=self.gpu_device, dtype=torch.uint8) for _ in range(self.cached_experts_num)]
        self.down_slots = [torch.empty(self.hidden_size * self.down_ggml_size, device=self.gpu_device, dtype=torch.uint8) for _ in range(self.cached_experts_num)]
        self.up_slots_ptr   = torch.tensor([slot.data_ptr() for slot in self.up_slots], dtype=torch.uint64, device=self.cpu_device)
        self.gate_slots_ptr = torch.tensor([slot.data_ptr() for slot in self.gate_slots], dtype=torch.uint64, device=self.cpu_device)
        self.down_slots_ptr = torch.tensor([slot.data_ptr() for slot in self.down_slots], dtype=torch.uint64, device=self.cpu_device)


        # print(self.gate_qtype, self.up_qtype, self.down_qtype)
        n_routed_experts = self.n_routed_experts
        self.cpu_infer = KExpertsCPU.CPU_INFER
        # n_routed_experts = len(self.orig_module)
        model_dtype = torch.get_default_dtype()
        if torch.xpu.is_available() and model_dtype == torch.float16:
            hidden_type = 1 # fp16
        else:
            hidden_type = 30 # bf16
        if self.backend == "llamafile":
            moe_config = MOEConfig(
                n_routed_experts,                  # expert_num
                self.config.num_experts_per_tok,   # routed_experts_num
                self.config.hidden_size,           # hidden_size
                self.config.moe_intermediate_size, # intermediate_size
                64,                                # stride
                10,                                # group_min_len
                1024,                              # group_max_len
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type,
                self.up_type,
                self.down_type,
                hidden_type, # TODO: get from model.dtype
            )
            self.moe = MOE(moe_config)
        elif self.backend == "AMXBF16":
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXBF16_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                max(cuda_graphs),
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = AMXBF16_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        elif self.backend == "AMXInt8":
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXInt8_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                max(cuda_graphs),
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = AMXInt8_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        num_experts_per_tok = self.config.num_experts_per_tok # 每次激活专家数量
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in KExpertsCPU.output_gpu_map:
            if isinstance(cuda_graphs, list):
                KExpertsCPU.output_gpu_map[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device=self.out_device) for i in range(len(cuda_graphs))]
            else:
                KExpertsCPU.output_gpu_map[self.out_device] = torch.zeros((cuda_graphs, self.config.hidden_size), device=self.out_device)
        if KExpertsCPU.input_tensor_cpu == None:
            if isinstance(cuda_graphs, list):
                KExpertsCPU.input_tensor_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.expert_ids_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.weights_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.output_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsCPU.bsz_tensor_cpu = [torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True) for i in range(len(cuda_graphs))]
            else:
                KExpertsCPU.input_tensor_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True)
                KExpertsCPU.expert_ids_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
                KExpertsCPU.weights_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
                if torch.xpu.is_available():
                    KExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True, dtype=model_dtype)
                    KExpertsCPU.bsz_tensor_cpu = torch.ones((1), device="cpu", dtype=torch.int32, pin_memory=True)
                else:
                    KExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
                    KExpertsCPU.bsz_tensor_cpu = torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True)
            
    def submit_for_one_decode(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if bsz_tensor is None:
            bsz_tensor = torch.ones(1, device=input_tensor.device, dtype=torch.int32)
        if cuda_graph_idx != -1:
            KExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
            KExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
            KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, 
                                                   self.moe.forward(1, 
                                                                    expert_ids.size(-1), 
                                                                    KExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), 
                                                                    KExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), 
                                                                    KExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), 
                                                                    KExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), 
                                                                    KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
        else:
            KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, 
                                                   self.moe.forward(1, 
                                                                    expert_ids.size(-1), 
                                                                    KExpertsCPU.expert_ids_cpu.data_ptr(), 
                                                                    KExpertsCPU.weights_cpu.data_ptr(), 
                                                                    KExpertsCPU.input_tensor_cpu.data_ptr(), 
                                                                    KExpertsCPU.output_cpu.data_ptr(), 
                                                                    KExpertsCPU.bsz_tensor_cpu.data_ptr()))


    def sync_for_one_decode(self, cuda_graph_idx=0):
        if cuda_graph_idx != -1:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]
        else:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device]
    
    
    @nvtx.annotate("KExpertsCPU.forward")
    def forward(self, mode, token_idx, input_tensor, expert_ids, weights, shared_experts = None, bsz_tensor=None, cuda_graph_idx=0, hit_rate=None, next_layer = None, timebreak=None):
        # assert next_layer is not None, "next_layer is None"
        identity = input_tensor
        expert_ids_gpu = expert_ids
        weights_gpu = weights
        input_tensor = input_tensor.view(-1, input_tensor.size(-1)) # reshape [batch_size * sequence_length, hidden_dim]  
        if bsz_tensor is None and (not torch.xpu.is_available() or input_tensor.size(0) > 1):
            bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32) # bsz_tensor = [batch_size * sequence_length]
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # 检测当前stream是否正在被cuda graph捕获
            if cuda_graph_idx != -1:
                KExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True) # copy input_tensor to cpu
                KExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
                KExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
                KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, 
                                                       self.moe.forward(expert_ids.size(0), 
                                                                        expert_ids.size(-1), 
                                                                        KExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), 
                                                                        KExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), 
                                                                        KExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), 
                                                                        KExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), 
                                                                        KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()
                                                                        )
                                                    )
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
                return KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]

            else:
                KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
                KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
                KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
                KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream,
                                                        self.moe.forward(expert_ids.size(0), 
                                                                         expert_ids.size(-1), 
                                                                         KExpertsCPU.expert_ids_cpu.data_ptr(), 
                                                                         KExpertsCPU.weights_cpu.data_ptr(), 
                                                                         KExpertsCPU.input_tensor_cpu.data_ptr(), 
                                                                         KExpertsCPU.output_cpu.data_ptr(), 
                                                                         KExpertsCPU.bsz_tensor_cpu.data_ptr()
                                                                         )
                                                    )
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
                return KExpertsCPU.output_gpu_map[self.out_device]
        elif input_tensor.size(0)==1 and torch.xpu.is_available():
            KExpertsCPU.input_tensor_cpu.copy_(input_tensor.view(-1), non_blocking=True)
            KExpertsCPU.expert_ids_cpu.copy_(expert_ids.view(-1), non_blocking=True)
            KExpertsCPU.weights_cpu.copy_(weights.view(-1), non_blocking=True)
            # KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor.view(-1), non_blocking=True)
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), KExpertsCPU.expert_ids_cpu.data_ptr(), KExpertsCPU.weights_cpu.data_ptr(), KExpertsCPU.input_tensor_cpu.data_ptr(), KExpertsCPU.output_cpu.data_ptr(), KExpertsCPU.bsz_tensor_cpu.data_ptr()))
            self.cpu_infer.sync()
            KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device].view(1, -1)
        else:
            if mode == "decode":
                if Config().prefetch_method == 0:
                    output = self.decode_with_token_prefetch(mode, token_idx, input_tensor, identity, expert_ids, weights, shared_experts, bsz_tensor, cuda_graph_idx, hit_rate, timebreak=timebreak)
                elif Config().prefetch_method == 1:
                    output = self.decode_with_layer_prefetch(mode, token_idx, input_tensor, identity, expert_ids, weights, shared_experts, bsz_tensor, cuda_graph_idx, hit_rate, next_layer, timebreak=timebreak)
                return output
            else:
                # non-generate mode, compute experts on CPU
                flat_idx = expert_ids.view(-1)
                batch_count = torch.bincount(flat_idx, minlength=256)
                self.expert_frequency += batch_count.cpu()
                # 初始化expert_cache
                top8_experts = torch.topk(self.expert_frequency, self.cached_experts_num).indices.to(self.cpu_device)
                
                # SMOE: expert cache load parameters
                for i in range(self.cached_experts_num):
                    up = self.gguf_loader.load_ggml_expert_from_weights(self.up, top8_experts[i], self.elements_per_expert, self.up_type)
                    gate = self.gguf_loader.load_ggml_expert_from_weights(self.gate, top8_experts[i], self.elements_per_expert, self.gate_type)
                    down = self.gguf_loader.load_ggml_expert_from_weights(self.down, top8_experts[i], self.elements_per_expert, self.down_type)

                    self.up_slots[i].copy_(up) # type: ignore
                    self.gate_slots[i].copy_(gate) # type: ignore
                    self.down_slots[i].copy_(down) # type: ignore

                    self.cached_experts["up_projs"][i].load(self.up_slots[i])
                    self.cached_experts["gate_projs"][i].load(self.gate_slots[i])
                    self.cached_experts["down_projs"][i].load(self.down_slots[i])
                self.cached_experts_ids = top8_experts.view(-1)
                self.cache_ready[0] = 1

                print(f"==++++++++++++>>>>. {self.key}: Expert cache initialized")
                

                in_gpu_mask = torch.zeros(self.n_routed_experts, dtype=torch.int64, device=self.cpu_device)
                input_tensor = input_tensor.contiguous().cpu()
                expert_ids = expert_ids.contiguous().cpu()
                weights = weights.contiguous().to(torch.float32).cpu()
                bsz_tensor = bsz_tensor.contiguous().cpu()
                output = torch.empty_like(input_tensor).contiguous()
                self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), 
                                                       expert_ids.size(1), 
                                                       expert_ids.data_ptr(), 
                                                       weights.data_ptr(), 
                                                       in_gpu_mask.data_ptr(), # 是否在GPU上
                                                       input_tensor.data_ptr(), 
                                                       output.data_ptr(), 
                                                       bsz_tensor.data_ptr()))
                y_ = shared_experts(identity)
                self.cpu_infer.sync()
                output = output.to(device=object.__getattribute__(self, "out_device")).view(identity.shape)
                output += y_
                return output
            
    @nvtx.annotate("KExpertsCPU.decode_with_token_prefetch")
    def decode_with_token_prefetch(self, mode, token_idx, input_tensor, identity, expert_ids, weights, shared_experts = None, bsz_tensor=None, cuda_graph_idx=0, hit_rate=None, timebreak=None):

        start_decode_time = time.time()
        for i in range(expert_ids.size(0)):
            for j in range(expert_ids.size(1)):
                self.expert_frequency[expert_ids[i][j]] += 1    
            
        gpu_compute = Config().gpu_compute
        in_gpu_mask = torch.zeros(self.n_routed_experts, dtype=torch.int64, device=self.cpu_device)
        if gpu_compute:
            for i in range(min(self.cached_experts_num, Config().gpu_compute_max_num)):
                in_gpu_mask[self.cached_experts_ids[i].item()] = 1

        input_tensor = input_tensor.contiguous().cpu()
        # input_tensor_32 = input_tensor.to(torch.float32)
        expert_ids = expert_ids.contiguous().cpu()
        weights = weights.contiguous().to(torch.float32).cpu()
        bsz_tensor = bsz_tensor.contiguous().cpu()
        output = torch.empty_like(input_tensor).contiguous()

        # 计算命中率
        mmask = torch.isin(expert_ids.squeeze(), self.cached_experts_ids)
        hn = mmask.sum()
        if hn > Config().gpu_compute_max_num:
            hn = torch.tensor(Config().gpu_compute_max_num)
        hn_rate = hn.item() / self.cached_experts_num
        hit_rate[self.layer_id].append(hn_rate)

        
        
        # torch.set_printoptions(precision=6, sci_mode=False)
        # print(f"传入数据bf16: {input_tensor[0,0:10]}")
        # print(f"python转换float32: {input_tensor_32[0,0:10]}")
        # cpu output
        overhead_time1 = time.time() - start_decode_time                                                                                     # overhead time before cpu start

        cpu_satrt_time = time.time()
        @nvtx.annotate("KExpertsCPU.cpu_commit", color="red")
        def cpu_commit():
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0),
                                                    expert_ids.size(1),
                                                    expert_ids.data_ptr(),
                                                    weights.data_ptr(),
                                                    in_gpu_mask.data_ptr(),  # 是否在GPU上
                                                    input_tensor.data_ptr(),
                                            output.data_ptr(), 
                                            bsz_tensor.data_ptr()))
        
        
        cpu_commit()
        
        
        # shared_expert output GPU
        shared_start = time.time()
        y_ = shared_experts(identity) # shape [batch_size, sequence_length, hidden_dim]
        shared_expert_time = time.time() - shared_start                                                                                             # shared expert time

        @nvtx.annotate("KExpertsCPU.waitfor", color="green")
        def wait_cache_ready():
            while self.cache_ready[0] != 1:
                continue
        
        if gpu_compute:
            wait_cache_start_time = time.time()
            wait_cache_ready()
            wait_cache_time = time.time() - wait_cache_start_time                                                                                # wait cache time

            gpu_compute_start_time = time.time()
            gpu_output = self.compute_gpu_experts_one(identity, expert_ids, weights, in_gpu_mask)
            gpu_compute_time = time.time() - gpu_compute_start_time                                                                              # gpu compute time

            # gpu_output = self.compute_gpu_experts_wrap(identity, expert_ids_gpu, weights_gpu)
            if Config().prefetch_num >= 0 and self.layer_id in KExpertsCPU.prefetch_layers:
                self.cache_ready[0] = 0
        

        
        # sync cpu and gpu
        @nvtx.annotate("KExpertsCPU.cpu_sync", color="red")
        def cpu_sync():
            self.cpu_infer.sync()
        cpu_sync()

        cpu_time = time.time() - cpu_satrt_time                                                                                                   # cpu compute time

        output = output.to(device=object.__getattribute__(self, "out_device")).view(identity.shape)

        # def batch_multihot_encode(y_idx_batch):
        #     batch_size = y_idx_batch.size(0)
        #     y = torch.zeros(batch_size, self.config.n_routed_experts)
        #     for i in range(batch_size):
        #         y[i][y_idx_batch[i]] = 1.0
        #     return y
        
        # 只有本层的prefetch开启时才开启预测和prefetch
        
        if self.layer_id in KExpertsCPU.prefetch_layers and gpu_compute:
            predict_start_time = time.time()
            if Config().prefetch_num >= 0:
                with torch.no_grad():
                    self.predictor = self.predictor.to(self.gpu_device)
                    # expert_ids_gpu = expert_ids.to(self.gpu_device)  # 确保expert_ids在GPU上
                    # expert_mask = batch_multihot_encode(expert_ids_gpu).to(self.gpu_device)  # shape [batch_size, expert_num]
                    # expert_mask = torch.zeros((expert_ids.size(0), self.n_routed_experts), device=self.gpu_device)
                    predicted_experts, probs = self.predictor.predict(identity.reshape(-1, identity.shape[2]))
                    self.predicted_experts_cpu = predicted_experts.to(self.cpu_device).view(-1)
            else:
                # 生成 0~255 的整数序列
                all_vals = torch.arange(256)
                perm = all_vals[torch.randperm(256)]
                self.predicted_experts_cpu = perm[:8]
            predict_time = time.time() - predict_start_time                                                                                       # predict time

            prefetch_start_time = time.time()
            @nvtx.annotate("KExpertsCPU.prefetch", color="blue")
            def sub_prefetch():
                self.cpu_infer.submit_prefetch(
                                            self.moe.prefetch(
                                                Config().prefetch_strategy,
                                                Config().prefetch_num,
                                                self.cached_experts_num,
                                                self.cached_experts_num,
                                                self.expert_frequency.data_ptr(), # expert频次统计
                                                self.predicted_experts_cpu.data_ptr(), # 新预测的expert的id
                                                self.cached_experts_ids.data_ptr(), # 当前cache的expert的id
                                                self.up_slots_ptr.data_ptr(), # up weight buffer
                                                self.gate_slots_ptr.data_ptr(), # gate weight buffer
                                                self.down_slots_ptr.data_ptr(), # down weight buffer
                                                self.cache_ready.data_ptr(), # 当前cache是否准备好
                                                KExpertsCPU.prefetch_stream.cuda_stream, # prefetch stream
                                            )
                                        )
            if Config().prefetch_num >= 0:
                sub_prefetch()
            prefetch_submit_time = time.time() - prefetch_start_time                                                                                     # prefetch time

        # get final output
        output += y_
        if gpu_compute:
            output += gpu_output
        

        end_decode_time = time.time()
        total_decode_time = end_decode_time - start_decode_time                                                                                     # total decode time


        # 统计时间结果
        timebreak['token_id'].append(token_idx)
        timebreak['layer_id'].append(self.layer_id)
        timebreak['hn_rate'].append(hn_rate)
        timebreak['overhead_time1'].append(overhead_time1)
        timebreak['cpu_time'].append(cpu_time)
        timebreak['shared_expert_time'].append(shared_expert_time)
        if gpu_compute:
            timebreak['wait_cache_time'].append(wait_cache_time)
            timebreak['gpu_compute_time'].append(gpu_compute_time)
        else:
            timebreak['wait_cache_time'].append(0.0)
            timebreak['gpu_compute_time'].append(0.0)
        if self.layer_id in KExpertsCPU.prefetch_layers and gpu_compute:
            timebreak['predict_time'].append(predict_time)
            timebreak['prefetch_submit_time'].append(prefetch_submit_time)
        else:
            timebreak['predict_time'].append(0.0)
            timebreak['prefetch_submit_time'].append(0.0)
        timebreak['total_decode_time'].append(total_decode_time)

        # if self.layer_id == 0:
        #     print(f"\nLayer {self.layer_id} token {token_idx} hit rate: {hn_rate:.4f}")
        #     print(output.shape)
        #     print(output.mean())
        #     print(output[0,0,0:10])
        #     print(gpu_output[0,0,0:10])
        # sys.exit(0)
        return output
    
    def decode_with_layer_prefetch(self, mode, token_idx, input_tensor, identity, expert_ids, weights, shared_experts = None, bsz_tensor=None, cuda_graph_idx=0, hit_rate=None, next_layer = None, timebreak=None):
        '''
        get next layer's gate to predict
        '''
        start_decode_time = time.time()
        for i in range(expert_ids.size(0)):
            for j in range(expert_ids.size(1)):
                self.expert_frequency[expert_ids[i][j]] += 1    
            
        gpu_compute = Config().gpu_compute
        in_gpu_mask = torch.zeros(self.n_routed_experts, dtype=torch.int64, device=self.cpu_device)# length: 256
        if gpu_compute:
            for i in range(min(self.cached_experts_num, Config().gpu_compute_max_num)):
                in_gpu_mask[self.cached_experts_ids[i].item()] = 1

        input_tensor = input_tensor.contiguous().cpu()
        expert_ids = expert_ids.contiguous().cpu()
        weights = weights.contiguous().to(torch.float32).cpu()
        bsz_tensor = bsz_tensor.contiguous().cpu()
        output = torch.empty_like(input_tensor).contiguous()

        # 计算命中率
        mmask = torch.isin(expert_ids.squeeze(), self.cached_experts_ids)
        hn = mmask.sum()
        if hn > Config().gpu_compute_max_num:
            hn = torch.tensor(Config().gpu_compute_max_num)
        hn_rate = hn.item() / self.cached_experts_num
        hit_rate[self.layer_id].append(hn_rate)

        overhead_time1 = time.time() - start_decode_time                                                                                     # overhead time before cpu start

        predict_start_time = time.time()
        if self.layer_id < 58 - Config().skip_layer:
            # print("-----------------flag0------------------")
            next_KDeepseekV3MoE = next_layer.mlp
            next_KExpertCPU = next_layer.mlp.experts.generate_experts
            # if self.layer_id in KExpertsCPU.prefetch_layers and gpu_compute:
            if gpu_compute:
                if Config().prefetch_num >= 0:
                    topk_ex, probs = next_KDeepseekV3MoE.gate(identity)
                    self.predicted_experts_cpu = topk_ex.to(self.cpu_device).view(-1)
                else:
                    # 生成 0~255 的整数序列
                    all_vals = torch.arange(256)
                    perm = all_vals[torch.randperm(256)]
                    self.predicted_experts_cpu = perm[:8]
                predict_time = time.time() - predict_start_time                                                                                       # predict time

                prefetch_submit_start = time.time()
                @nvtx.annotate("KExpertsCPU.prefetch", color="blue")
                def sub_prefetch():
                    self.cpu_infer.submit_prefetch(
                                                next_KExpertCPU.moe.prefetch(
                                                    Config().prefetch_strategy,
                                                    Config().prefetch_num,
                                                    self.cached_experts_num,
                                                    self.cached_experts_num,
                                                    next_KExpertCPU.expert_frequency.data_ptr(), # expert频次统计
                                                    self.predicted_experts_cpu.data_ptr(), # 新预测的expert的id
                                                    next_KExpertCPU.cached_experts_ids.data_ptr(), # 当前cache的expert的id
                                                    next_KExpertCPU.up_slots_ptr.data_ptr(), # up weight buffer
                                                    next_KExpertCPU.gate_slots_ptr.data_ptr(), # gate weight buffer
                                                    next_KExpertCPU.down_slots_ptr.data_ptr(), # down weight buffer
                                                    next_KExpertCPU.layer_prefetch_ready.data_ptr(), # 当前cache是否准备好
                                                    KExpertsCPU.prefetch_stream.cuda_stream, # prefetch stream
                                                )
                                            )
                # print("-----------------flag1------------------")
                if Config().prefetch_num >= 0:  
                    # print(f"submit {next_KExpertCPU.layer_id} prefetch")
                    sub_prefetch()
                prefetch_submit_time = time.time() - prefetch_submit_start                                                                                     # prefetch time
        # cpu output
        cpu_start_time = time.time()
        @nvtx.annotate("KExpertsCPU.cpu_commit", color="red")
        def cpu_commit():
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0),
                                                    expert_ids.size(1),
                                                    expert_ids.data_ptr(),
                                                    weights.data_ptr(),
                                                    in_gpu_mask.data_ptr(),  # 是否在GPU上
                                                    input_tensor.data_ptr(),
                                            output.data_ptr(), 
                                            bsz_tensor.data_ptr()))
        cpu_commit()
        
        
        # shared_expert output GPU
        shared_expert_start_time = time.time()
        y_ = shared_experts(identity) # shape [batch_size, sequence_length, hidden_dim]
        shared_expert_time = time.time() - shared_expert_start_time                                                                                               # shared_expert_time
        
        @nvtx.annotate("KExpertsCPU.waitfor", color="green")
        def wait_cache_ready():
            while self.layer_prefetch_ready[0] != 1:
                continue
        
        if gpu_compute:
            if self.layer_id < Config().skip_layer:
                self.layer_prefetch_ready[0] = 1
            if Config().prefetch_num <= 0:
                self.layer_prefetch_ready[0] = 1
            # if self.layer_id == 57:
                # print(f"\nbefore wait flag: {self.layer_prefetch_ready}")
            # print(f"token {token_idx}, layer {self.layer_id}, waitflag: {self.layer_prefetch_ready}")
            wait_cache_ready_start_time = time.time()
            wait_cache_ready()
            wait_cache_time = time.time() - wait_cache_ready_start_time                                                                                # wait cache time
            # if self.layer_id == 57:
                # print(f"\nlayer {self.layer_id}, token {token_idx}, self.cached_experts_ids: {self.cached_experts_ids}")
            gpu_compute_start_time = time.time()
            gpu_output = self.compute_gpu_experts_one(identity, expert_ids, weights, in_gpu_mask)
            gpu_compute_time = time.time() - gpu_compute_start_time                                                                              # gpu compute time
            self.layer_prefetch_ready[0] = 0
        
        # sync cpu and gpu
        @nvtx.annotate("KExpertsCPU.cpu_sync", color="red")
        def cpu_sync():
            self.cpu_infer.sync()
        cpu_sync()
        cpu_time = time.time() - cpu_start_time                                                                                                   # cpu compute time
        output = output.to(device=object.__getattribute__(self, "out_device")).view(identity.shape)

        # get final output
        output += y_
        if gpu_compute:
            output += gpu_output

        total_decode_time = time.time() - start_decode_time                                                                                     # total decode time

        # 统计时间结果
        timebreak['token_id'].append(token_idx)
        timebreak['layer_id'].append(self.layer_id)
        timebreak['hn_rate'].append(hn_rate)
        timebreak['overhead_time1'].append(overhead_time1)
        timebreak['cpu_time'].append(cpu_time)
        timebreak['shared_expert_time'].append(shared_expert_time)
        if gpu_compute and self.layer_id < 58 - Config().skip_layer:
            timebreak['wait_cache_time'].append(wait_cache_time)
            timebreak['gpu_compute_time'].append(gpu_compute_time)
        else:
            timebreak['wait_cache_time'].append(0.0)
            timebreak['gpu_compute_time'].append(0.0)
        if self.layer_id in KExpertsCPU.prefetch_layers and gpu_compute and self.layer_id < 58 - Config().skip_layer:
            timebreak['predict_time'].append(predict_time)
            timebreak['prefetch_submit_time'].append(prefetch_submit_time)
        else:
            timebreak['predict_time'].append(0.0)
            timebreak['prefetch_submit_time'].append(0.0)
        timebreak['total_decode_time'].append(total_decode_time)
        

        return output
        
    @nvtx.annotate("KExpertsCPU.compute_gpu_experts")
    def compute_gpu_experts(self, input_tensor, expert_ids, weights):

        input_tensor = input_tensor.to(self.gpu_device)  # 确保输入张量在GPU上
        expert_ids = expert_ids.to(self.gpu_device)  # 确保expert_ids在GPU上
        weights = weights.to(self.gpu_device)  # 确保weights在GPU上
        # print(f"shapes: {input_tensor.shape}, {expert_ids.shape}, {weights.shape}")

        # 初始化加权输出
        weighted_output = torch.zeros_like(input_tensor, device=self.gpu_device, dtype=input_tensor.dtype)  # [batch_size, sequence_length, hidden_dim]

        # 找出需要处理的专家中哪些已缓存到 GPU 上
        unique_expert_ids = torch.unique(expert_ids).to(self.cpu_device)
        gpu_compute_expert_ids = unique_expert_ids[torch.isin(unique_expert_ids, self.cached_experts_ids)]

        for expert_id in gpu_compute_expert_ids:

            # 找出 expert_id 的在 batch 中对应位置
            mask = (expert_ids == expert_id) # [batch_size, num_experts_per_tok]
            idx_b, idx_e = torch.nonzero(mask, as_tuple=True) # 正常返回一个坐标，这里将其转换为两个索引张量, 长度都为该 expert_id 在 batch 中出现的次数

            # 提取对应的权重并扩展用于广播
            selected_weights = weights[idx_b, idx_e].view(-1, 1, 1)  # [N, 1, 1]

            # 提取对应的输入: [N, sequence_length, hidden_dim]
            selected_inputs = input_tensor[idx_b, :, :] 

            # 找到 cached_experts 中对应位置
            local_idx = (self.cached_experts_ids == expert_id).nonzero(as_tuple=True)[0]
            gate_proj = self.cached_experts["gate_projs"][local_idx]
            up_proj = self.cached_experts["up_projs"][local_idx]
            down_proj = self.cached_experts["down_projs"][local_idx]

            # 计算 expert_compute_output
            # print(self.gate_type)
            gated = gate_proj(selected_inputs, self.gate_type)              # [N, sequence_length, hidden_dim]
            upped = up_proj(selected_inputs, self.up_type)               # [N, sequence_length, hidden_dim]
            activated = self.act_fn(gated) * upped         # [N, sequence_length, hidden_dim]
            downed = down_proj(activated, self.down_type)                  # [N, sequence_length, hidden_dim]
            # print(f"gated type: {gated.dtype}, upped type: {upped.dtype}, downed type: {downed.dtype}")
            # print(f"weights type: {selected_weights.dtype}, input type: {selected_inputs.dtype}")

            # 权重加权
            downed_weighted = downed * selected_weights    # [N, sequence_length, hidden_dim]
            downed_weighted = downed_weighted.to(input_tensor.dtype)  # 确保类型一致
            # print(f"down weight type: {downed_weighted.dtype}, output type: {weighted_output.dtype}")

            # Scatter Add 回到 weighted_output 对应位置
            weighted_output.index_add_(0, idx_b, downed_weighted)

        return weighted_output
    
    @nvtx.annotate("KExpertsCPU.compute_gpu_experts_one")
    def compute_gpu_experts_one(self, input_tensor, expert_ids, weights, in_gpu_mask=None):
        '''
        input_tensor: [1, 1, hidden_dim]
        expert_ids: [1, 8]
        weights: [1, 8]
        '''
        # print(f"input tenser dtype: {input_tensor.dtype}")
        # sys.exit(0)
        input_tensor = input_tensor.to(self.gpu_device)  # 确保输入张量在GPU上
        expert_ids = expert_ids.to(self.gpu_device)  # 确保expert_ids在GPU上
        weights = weights.to(self.gpu_device)  # 确保weights在GPU上
        # print(f"weights shape: {weights.shape}")

        weighted_output = torch.zeros_like(input_tensor, device=self.gpu_device, dtype=input_tensor.dtype)  # [batch_size, sequence_length, hidden_dim]

        for i in range(self.cached_experts_num):
            if self.cached_experts_ids[i] not in expert_ids or in_gpu_mask[self.cached_experts_ids[i]] == 0:
                continue
            else:
            # if self.cached_experts_ids[i] in expert_ids:
                gate_proj = self.cached_experts["gate_projs"][i]
                up_proj = self.cached_experts["up_projs"][i]
                down_proj = self.cached_experts["down_projs"][i]
                gated = gate_proj(input_tensor, self.gate_type)
                upped = up_proj(input_tensor, self.up_type)
                activated = self.act_fn(gated) * upped
                downed = down_proj(activated, self.down_type)

                weight = weights[expert_ids == self.cached_experts_ids[i]].view(-1, 1, 1)  # [N, 1, 1]
                weighted = downed * weight  # [N, sequence_length, hidden_dim]
                weighted_output = weighted_output + weighted

        return weighted_output
    
    @nvtx.annotate("KExpertsCPU.compute_gpu_experts_wrap")
    def compute_gpu_experts_wrap(self, input_tensor, expert_ids, weights):
        # print(input_tensor.shape, expert_ids.shape, weights.shape)
        cached_ids_gpu = self.cached_experts_ids.to(self.gpu_device)
        return self.cached_experts_wrap(input_tensor, expert_ids, weights, cached_ids_gpu, 
                                        self.gate_slots, self.up_slots, self.down_slots, 
                                        self.gate_type, self.up_type, self.down_type, 
                                        self.act_fn)
    
    @nvtx.annotate("KExpertsCPU.prefetch_experts")
    def prefetch_experts(self, token_idx, input_tensor, expert_ids):
        """
        Predict next experts and update expert cache efficiently:
        - 统计 predicted_experts 出现频次，按频次从高到低选8个专家
        - 仅在新专家未在 cache 中时替换（减少数据传输）
        - 替换时保持 cached_experts 中顺序不变，仅在可替换位置更新
        - 同步更新 self.cached_experts_ids
        # 未使用
        """
        if self.layer_id == self.print_layer:
            print(f"before prefetch flag {self.is_prefetch_done}")
        # torch.cuda.set_device(self.gpu_device)
        def batch_multihot_encode(y_idx_batch):
            batch_size = y_idx_batch.size(0)
            y = torch.zeros(batch_size, self.config.n_routed_experts)
            for i in range(batch_size):
                y[i][y_idx_batch[i]] = 1.0
            return y
        self.predictor = self.predictor.to(self.gpu_device)
        input_tensor = input_tensor.to(self.gpu_device)  # 确保输入张量在GPU上
        expert_ids = expert_ids.to(self.gpu_device)  # 确保expert_ids在GPU上
        # 将expert_ids转换为multihot编码 (B, E) -> (B, 256)
        expert_mask = batch_multihot_encode(expert_ids).to(self.gpu_device)  # shape [batch_size, expert_num]
        with torch.cuda.stream(KExpertsCPU.prefetch_stream):
            predicted_experts, probs = self.predictor.predict(input_tensor.reshape(-1, input_tensor.shape[2]), expert_mask=expert_mask)
            # print(f"predicted:          {predicted_experts}")
            predicted_flat = predicted_experts.flatten()

            # 1) 统计出现频次
            unique_experts, counts = torch.unique(predicted_flat, return_counts=True)
            sorted_idx = torch.argsort(counts, descending=True)
            top_experts = unique_experts[sorted_idx][:self.cached_experts_num]

            # 2) 找出 top_experts 中不在 cache 中的需要加载的新专家
            need_load_mask = ~torch.isin(top_experts, self.cached_experts_ids)
            need_load_ids = top_experts[need_load_mask]

            # 3) 找出 cache 中可被替换的位置（不在 top_experts 中的）
            can_replace_mask = ~torch.isin(self.cached_experts_ids, top_experts)
            can_replace_indices = torch.nonzero(can_replace_mask).flatten()

            replace_num = min(len(need_load_ids), len(can_replace_indices))

            if self.layer_id == self.print_layer:
                print(f"Token {token_idx}, layer {self.layer_id}, cache {self.cached_experts_ids}, predicted {predicted_experts}, need prefetch {replace_num} experts")

            if replace_num > 0:
                replace_ids = need_load_ids[:replace_num].to(self.cpu_device)
                replace_indices = can_replace_indices[:replace_num]

                for idx_tensor, new_id_tensor in zip(replace_indices, replace_ids):
                    idx = idx_tensor.item()
                    new_id = new_id_tensor.item()

                    up = self.gguf_loader.load_ggml_expert_from_weights(self.up, new_id, self.elements_per_expert, self.up_type)
                    # up = up.to(self.gpu_device, non_blocking=True)
                    gate = self.gguf_loader.load_ggml_expert_from_weights(self.gate, new_id, self.elements_per_expert, self.gate_type)
                    # gate = gate.to(self.gpu_device, non_blocking=True)
                    down = self.gguf_loader.load_ggml_expert_from_weights(self.down, new_id, self.elements_per_expert, self.down_type)
                    # down = down.to(self.gpu_device, non_blocking=True)

                    self.up_slots[idx].copy_(up) # type: ignore
                    self.gate_slots[idx].copy_(gate) # type: ignore
                    self.down_slots[idx].copy_(down) # type: ignore

                    self.cached_experts["up_projs"][idx].load(self.up_slots[idx])
                    self.cached_experts["gate_projs"][idx].load(self.gate_slots[idx])
                    self.cached_experts["down_projs"][idx].load(self.down_slots[idx])

                    # 更新缓存 ID
                    self.cached_experts_ids[idx] = new_id
            # print(f"after prefetch:     {self.cached_experts_ids}")
            # cuda event
            # if self.layer_id == self.print_layer:
            #     print(f"beToken {token_idx}, layer {self.layer_id}, prefetch done! event {self.prefetch_event.query()}")
            # self.prefetch_event.record(KExpertsCPU.prefetch_stream)
            self.is_prefetch_done = True
            if self.layer_id == self.print_layer:
                print(f"Token {token_idx}, layer {self.layer_id}, prefetch done! flag {self.is_prefetch_done}")

    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        # TODO: support Bias
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key] #...experts

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                # gate = self.gguf_loader.load_gguf_tensor(key + ".ffn_gate_exps.weight")
                # up = self.gguf_loader.load_gguf_tensor(key + ".ffn_up_exps.weight")
                # down = self.gguf_loader.load_gguf_tensor(key + ".ffn_down_exps.weight")
                # gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                # up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                # down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
                gate_type = self.gguf_loader.get_ggml_type(key + ".ffn_gate_exps.weight")
                up_type = self.gguf_loader.get_ggml_type(key + ".ffn_up_exps.weight")
                down_type = self.gguf_loader.get_ggml_type(key + ".ffn_down_exps.weight")
            
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_gate.{i}.weight")
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_down.{i}.weight")
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.get_ggml_type(key + ".ffn_gate.0.weight")
                up_type = self.gguf_loader.get_ggml_type(key + ".ffn_up.0.weight")
                down_type = self.gguf_loader.get_ggml_type(key + ".ffn_down.0.weight")
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
class KExpertsMarlin(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size

        # create empty marlin experts according to the number of experts per token
        # up
        self.up_projs = [KLinearMarlin(key+ "." + "ffn_up_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # gate
        self.gate_projs = [KLinearMarlin(key+ "." + "ffn_gate_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # down
        self.down_projs = [KLinearMarlin(key+ "." + "ffn_down_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        if w is None:
            w = self.load_weights()
            load_by_experts = True

        if load_by_experts:
            if isinstance(w, dict):
                self.gate = w["gate"]
                self.up = (w["up"])
                self.down = (w["down"])
                for i in tqdm(range(self.expert_num), desc=f"Dequanting and quanting for KExpertsMarlin {self.key}"):
                    up_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_up_exps.weight", self.up, i, self.elements_per_tensor, device=self.device)
                    gate_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_gate_exps.weight", self.gate, i, self.elements_per_tensor, device=self.device)
                    down_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_down_exps.weight", self.down, i, self.elements_per_tensor, device=self.device)
                    
                    self.up_projs[i].load(nn.Parameter(up_weights), device=device)
                    self.gate_projs[i].load(nn.Parameter(gate_weights), device=device)
                    self.down_projs[i].load(nn.Parameter(down_weights), device=device)
                    self.loaded_experts_idx.append(i)
        else:
            if isinstance(w, dict):
                self.gate = w["gate"]
                self.up = (w["up"])
                self.down = (w["down"])
                for i in range(self.expert_num):
                    self.up_projs[i].load(nn.Parameter(self.up[i,...]), device=device)
                    self.gate_projs[i].load(nn.Parameter(self.gate[i,...]), device=device)
                    self.down_projs[i].load(nn.Parameter(self.down[i,...]), device=device)
                    self.loaded_experts_idx.append(i)
        return 

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    @nvtx.annotate("KExpertsMarlin.forward")
    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        org_dtype = hidden_states_cpu.dtype
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device).to(org_dtype)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.expert_num):
            if not expert_mask[expert_idx].any():
                continue
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = self.gate_projs[expert_idx].forward(current_state)
            A = self.act_fn(G)
            U = self.up_projs[expert_idx].forward(current_state)
            H = A * U  # Element-wise multiplication
            current_hidden_states = self.down_projs[expert_idx].forward(H) * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        
        return final_hidden_states.to(dtype=org_dtype, device=org_device)

# untested, CUDA OOM
class KExpertsTorch(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        # self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size
        self.gate = [None for _ in range(self.expert_num)]
        self.up = [None for _ in range(self.expert_num)]
        self.down = [None for _ in range(self.expert_num)]
        self.dtype = torch.get_default_dtype()

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        if w is None:
            w = self.load_weights()
            load_by_experts = True

        if load_by_experts:
            if isinstance(w, dict):
                for i in tqdm(range(self.expert_num), desc=f"Dequanting for KExpertsTorch {self.key}"):
                    up_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_up_exps.weight", w["up"], i, self.elements_per_tensor, device=self.device)
                    gate_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_gate_exps.weight", w["gate"], i, self.elements_per_tensor, device=self.device)
                    down_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_down_exps.weight", w["down"], i, self.elements_per_tensor, device=self.device)
                    
                    self.up[i] = up_weights
                    self.gate[i] = gate_weights
                    self.down[i] = down_weights
        else:
            if isinstance(w, dict):
                for i in range(self.expert_num):
                    self.gate[i] = w["gate"][i, ...].to(device=device, dtype=self.dtype)
                    self.up[i] = w["up"][i, ...].to(device=device, dtype=self.dtype)
                    self.down[i] = w["down"][i, ...].to(device=device, dtype=self.dtype)
        
        self.up = torch.stack(self.up, dim=0)
        self.gate = torch.stack(self.gate, dim=0)
        self.down = torch.stack(self.down, dim=0)
        return 

    def unload(self):
        if self.gate is not None:
            self.gate = None
            self.up = None
            self.down = None

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:

        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=self.gate.dtype, device=hidden_states_cpu.device
        )
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.expert_num):
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = current_state @ self.gate[expert_idx,...].T
            A = self.act_fn(G)
            U = current_state @ self.up[expert_idx,...].T
            H = A * U  # Element-wise multiplication
            current_hidden_states = H @ self.down[expert_idx,...].T * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)


        return final_hidden_states.to(dtype=org_dtype, device=org_device)

EXPERTS_MAP = {
    "KExpertsCPU": KExpertsCPU,
    "KExpertsTorch": KExpertsTorch,
    "KExpertsMarlin": KExpertsMarlin,
}

class KTransformersExperts(BaseInjectedModule, KExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                #  device: str = "cuda",
                 prefill_device:str = "cuda",
                 prefill_op: str | None = "KExpertsTorch",
                 generate_device: str = "cpu",
                 generate_op: str | None = "KExpertsCPU",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, 
                                                             gguf_loader, 
                                                             config, 
                                                             len(orig_module), 
                                                             device=generate_device,  
                                                             **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        # print(f"----------------------> Loading experts in {mode} mode, from KTransformersExperts")
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    @nvtx.annotate("KTransformersExperts.forward")
    def forward(self, mode, token_idx, input_tensor, expert_ids, weights, shared_experts=None, hit_rate=None, next_layer = None, timebreak=None):
        # assert next_layer is not None, "next_layer is None"
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            # assert next_layer is not None, "next_layer is None"
            return self.generate_experts.forward(mode, token_idx, input_tensor, expert_ids, weights, shared_experts=shared_experts, hit_rate=hit_rate, next_layer=next_layer, timebreak=timebreak)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")


from ktransformers.models.modeling_deepseek import DeepseekV2MoE
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MoE
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from ktransformers.models.modeling_mixtral import MixtralSparseMoeBlock


class KQwen2MoeSparseMoeBlock(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += shared_expert_output
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights.cpu()

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

class KDeepseekV2MoE(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KDeepseekV3MoE(BaseInjectedModule, DeepseekV3MoE):
    layer_counter = 0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_id = KDeepseekV3MoE.layer_counter
        KDeepseekV3MoE.layer_counter += 1
        
    
    def record_topk_idx(self, prompt_name, mode, token_idx, layer_idx, topk_idx, hidden_states):
        import json
        if mode == "decode":
            # directory = os.path.join("./", "topk_idx")
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            file = prompt_name + ".json"
        
            record = {
                "mode": mode,
                "token_idx": token_idx,
                "layer_idx": layer_idx,
                "topk_idx": topk_idx.tolist(),
                "hidden_states": hidden_states.tolist(),
            }
            with open(file, "a") as f:
                json.dump(record, f)
                f.write("\n")
    
    @nvtx.annotate("KDeepseekV3MoE.forward")
    def forward(self, hidden_states, prompt_name, mode, token_idx, hit_rate, next_layer, timebreak):
        # assert next_layer is not None, "next_layer is None"
        # print(f"===>>>>    token_idx:{token_idx}")
        identity = hidden_states # shape [batch_size, sequence_length, hidden_dim]
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        # hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) # shape [batch_size * sequence_length, hidden_dim]

        # if prompt_name is None:
        #     sys.exit("prompt_name is None, please set it to a valid value")
        # self.record_topk_idx(prompt_name, mode, token_idx, self.layer_id, topk_idx, hidden_states)
        
        
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0], self.layer_id)
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        # for prefill phase
        # shared experts直接计算
        # if self.config.n_shared_experts is not None:
        #     y_ = self.shared_experts(identity).squeeze(0)

        # routed experts 如果是KTrans的实现，则进入experts的forward方法  
        # 修改：将shared_experts的参数传入routed experts的forward方法实现并行计算  
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(mode, token_idx, hidden_states, topk_idx, topk_weight, shared_experts=self.shared_experts, hit_rate = hit_rate, next_layer = next_layer, timebreak=timebreak).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        # if self.config.n_shared_experts is not None:
        #     y += y_
        return y



    @torch.no_grad()
    def moe_kexperts(self, mode, token_idx, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, shared_experts = None, hit_rate = None, next_layer = None, timebreak = None) -> torch.Tensor:
        outs = self.experts(mode, token_idx, x, topk_ids, topk_weight, shared_experts, hit_rate, next_layer, timebreak)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KMistralSparseMoEBlock(BaseInjectedModule, MixtralSparseMoeBlock):
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states_expert.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts_expert.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights_expert.cpu()

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
            
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

class KDeepseekV3MoEV2(BaseInjectedModule, DeepseekV3MoE):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx)
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KTransformersExpertsV2(BaseInjectedModule, KExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                #  device: str = "cuda",
                 prefill_device:str = "cuda",
                 prefill_op: str | None = "KExpertsTorch",
                 generate_device: str = "cpu",
                 generate_op: str | None = "KExpertsCPU",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx=0):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

class KQwen2MoeSparseMoeBlockV2(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states, bsz_tensor)        

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx)
            y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
            y_ = F.sigmoid(self.shared_expert_gate(hidden_states)) * y_    

            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            
            y += y_
            y.resize_(*orig_shape)
            return y

        y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
        y_ = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * y_
        )


        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            ) 
        y += y_
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KQwen3MoeSparseMoeBlockV2(BaseInjectedModule, Qwen3MoeSparseMoeBlock):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states, bsz_tensor)        

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx)
            # y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
            # y_ = F.sigmoid(self.shared_expert_gate(hidden_states)) * y_    

            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            
            # y += y_
            y.resize_(*orig_shape)
            return y

        # y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
        # y_ = (
        #     F.sigmoid(self.shared_expert_gate(hidden_states)) * y_
        # )


        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            ) 
        # y += y_
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out