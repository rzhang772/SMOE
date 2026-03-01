/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <iostream>
#include <cstdint>
#include <cassert>
#include <thread>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <unordered_set>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

MOE::MOE(MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;
    
    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);
    size_t exp_inter_hidden_mul_ = (size_t)config.expert_num * config.intermediate_size * config.hidden_size;
    for (int i = 0; i < numa_nodes; i++) {
        gate_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type), i);
        up_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type), i);
        down_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type), i);
        if (!gate_proj_numa_[i]) {
            std::cout << "Memory allocation failed for gate_proj_numa_ on node " << i << std::endl;
        }
        if (!up_proj_numa_[i]) {
            std::cout << "Memory allocation failed for up_proj_numa_ on node " << i << std::endl;
        }
        if (!down_proj_numa_[i]) {
            std::cout << "Memory allocation failed for down_proj_numa_ on node " << i << std::endl;
        }
        memcpy(gate_proj_numa_[i], gate_proj_, exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type));
        memcpy(up_proj_numa_[i], up_proj_, exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type));
        memcpy(down_proj_numa_[i], down_proj_, exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type));
    }
    #endif

    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    s_mem_requests.push_back({(void**)&s_up_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
    shared_mem_buffer.alloc(this, s_mem_requests);

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    gate_nbytes = gate_bytes();
    up_nbytes   = up_bytes();
    down_nbytes = down_bytes();
    
    // printf("MOE::MOE, up_nbytes: %zu, gate_nbytes: %zu, down_nbytes: %zu\n", up_nbytes, gate_nbytes, down_nbytes);
    cudaError_t err;
    err = cudaMallocHost(&up_proj_pinned,   up_nbytes);
    if (err != cudaSuccess) {
    printf("cudaMallocHost failed: %s\n", cudaGetErrorString(err));
}
    cudaMallocHost(&gate_proj_pinned, gate_nbytes);
    cudaMallocHost(&down_proj_pinned, down_nbytes);
}

MOE::~MOE() {
    shared_mem_buffer.dealloc(this);

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    for (int i = 0; i < numa_nodes; i++) {
        numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
    }
    #endif

    // moe_free_pinned(up_proj_pinned, gate_proj_pinned, down_proj_pinned);
    if (up_proj_pinned)   cudaFreeHost(up_proj_pinned);
    if (gate_proj_pinned) cudaFreeHost(gate_proj_pinned);
    if (down_proj_pinned) cudaFreeHost(down_proj_pinned);
}

void MOE::warm_up(KBackend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, 
            nullptr, 
            input.data(), output.data(), backend);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}


/**
forward one总结：
一次只有一个token，即
expert_ids: list, 长度k
weights: list, 长度k
input: [1, hidden_size]，输入的token
output: [1, hidden_size]，输出的token

1. 首先判断input数据类型是不是计算用的类型，如果不是则转换为计算用的类型 fp32，然后让gate_inout_ptr和up_input_ptr指向input
2. 计算gate_proj @ input和up_proj @ input和激活函数，得到gate_output_ptr和up_output_ptr。并行逻辑为将gate_proj和up_proj矩阵按列切成若干块，每块为[hidden_size, stride]
3. 计算down_proj, 并行逻辑为将down_proj矩阵按列切成若干块，每块为[intermediate_size, stride]，得到down_output_ptr并进行加权

*/
void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights,
     const uint64_t* in_gpu_mask, 
     const void* input, void* output, KBackend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;

    // if (in_gpu_mask != nullptr){
    //     for(int i=0;i<256;i++){
    //         std::cout<<in_gpu_mask[i]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    

    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        // printf("===============================>>>>>>MOE::forward_one: input data type != gate_type or up_type, need conversion to fp32\n");
        // printf("config_.hidden_type: %d, gate_type: %d, up_type: %d\n", config_.hidden_type, config_.gate_type, config_.up_type);

        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type); // input -> s_input_fp32_
        // printf("C++转换:\n");
        // for(int iii=0;iii<10;iii++){
        //     printf("s_input_fp32_[%d]: %f\n", iii, s_input_fp32_[iii]);
        // }
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type); // s_input_fp32_ -> s_gate_input_
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }

    int nth = config_.intermediate_size / config_.stride; // stride = 64, 2048/64=32, 32*8=256 tasks， 即每个矩阵按照列切成32块
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];

        // printf("====================in forward_one() gate and up before gpu mask====================\n");
        // 检查是否在 GPU 上, 如果在 GPU 上则跳过计算
        if(in_gpu_mask != nullptr) {
            // Use uint64_t for in_gpu_mask to match expert_ids
            // This assumes in_gpu_mask is a 1D array where each entry corresponds to an expert_id
            // If the expert_id is in GPU, skip the computation
            // Note: Ensure that in_gpu_mask is properly initialized and passed
            // std::cout << "expert_id: " << expert_id << ", in_gpu_mask[expert_id]: " << in_gpu_mask[expert_id] << std::endl;
            if (in_gpu_mask[expert_id]) return;
        }

        int ith = task_id % nth; // 第i块/32块
        
        // gate_proj_: [256*2048*4032] config_.hidden_size7168 * ggml_type_size(config_.up_type)144 / ggml_blck_size(config_.up_type)256 = 4032
        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[KBackend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ 
                            + (expert_id * config_.intermediate_size + ith * config_.stride) 
                                * config_.hidden_size 
                                * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        // 计算 gate @ input
        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride; // 第idx个expert第ith块的输出
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        // printf("====================in forward_one() after gate====================\n");
        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[KBackend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        // 计算 up @ input
        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        // printf("====================in forward_one() after up====================\n");
        
        // 计算 intermediate_fp32 = act_fn(gate_output) * up_output
        // 这一块中每一个列向量计算结果的输出值做计算 -》 s_intermediate_fp32_ shape [k, 2048]
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }

        // 将 intermediate_fp32 转换为 down_input
        // 如果块大小可以支持按块移动
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            // 本块的计算结果地址
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            // 对应在down_input中的存放位置
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            // 转换为对应数据类型并存放到对应位置
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);

    // 块大小不支持按块移动，则以expert为单位移动
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }

    // compute down, original shape [2048, 7168, 256] -> np.dims [256, 7168, 1152] Q4_k
    // config_.intermediate_size2048 * ggml_type_size(config_.down_type)144 / ggml_blck_size(config_.down_type)256 = 1152
    nth = config_.hidden_size / config_.stride; // nth = 7168/64 = 112
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            if (in_gpu_mask != nullptr) {
                // Use uint64_t for in_gpu_mask to match expert_ids
                // This assumes in_gpu_mask is a 1D array where each entry corresponds to an expert_id
                // If the expert_id is in GPU, skip the computation
                // Note: Ensure that in_gpu_mask is properly initialized and passed
                if (in_gpu_mask[expert_id]) continue;
            }

            #ifdef USE_NUMA
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[KBackend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #else
            // 定位到expert_id的down_proj_的第ith块的起始位置
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            // 获取到该块在本次也就是down@input输出中的位置, 计算的结果输出到该指针指向的s_down_output_的对应位置
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            
            // 为本块的结果进行加权
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        // 将本块的输出写到最终输出中，如果块大小允许
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);

    // 如果块大小不允许按块移动，则以expert为单位移动
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    }
}

/**
forward_many总结：
并行逻辑与forward_one相同，不同点在于：
1. forward_one只处理一个token，因此每个矩阵乘法的输入大小都为[1, hidden_size]，
    而forward_many处理多个token，因此每个矩阵乘法的输入大小为[number of tokens, hidden_size]，这里的number of tokens为该expert需要处理的token数量
2. forward_one的stride=64，也就是每个矩阵切分为2048/64=32块，而forward_many的stride=256，也就是每个矩阵切分为2048/256=8块，每个线程处理的数据量更大
 */

// 计算多个token，>10时
void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, 
    const uint64_t* in_gpu_mask, 
    const void* input, void* output, KBackend* backend) {
    // 统计每个expert的处理的token数量， 所有256个都统计，而不是只统计在expert_ids中的
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    // m_local_pos_: shape[qlen, k] same with expert_ids, 每个token在每个expert计算中的位置，即该token[i]是在该expert[j]计算中的第几个token
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }

    // 计算每个expert的在分配的内存中的地址，offset为每个expert处理多少个token
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        // 所有expert与gate和up计算的输入输出
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        // 临时存储fp32格式的输出
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        // 所有expert与down计算的输入输出
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }

    // 将 token i 的 input(hidden_state) 分发到每个token-i激活的expert的 gate_input 和 up_input 中
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        // 将 token i 的 input(hidden_state) 分发到每个token-i激活的expert的 gate_input 和 up_input 中
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type),
                gate_input_ptr, 
                config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), 
                up_input_ptr, 
                config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);

    int stride = QK_K; // ggml-quants -> ggml-common = 256
    int nth = config_.intermediate_size / stride; // 2048/256 = 8, 8*256 = 2048 tasks
    // 计算 gate @ input 和 up @ input， 将gate和up矩阵按列切块即2048/256=8块，每块为[7168,256], 
    // 每个expert由8个线程处理，每个处理[B, 7168]@[7168, 256] -> [B, 256],最后拼接
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth; // 第几个expert
        int ith = task_id % nth;// 该expert的第ith块
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];
        // printf("====================in forward_many() gate and up before gpu mask====================\n");
        if(in_gpu_mask != nullptr) {
            // Use uint64_t for in_gpu_mask to match expert_ids
            // This assumes in_gpu_mask is a 1D array where each entry corresponds to an expert_id
            // If the expert_id is in GPU, skip the computation
            // Note: Ensure that in_gpu_mask is properly initialized and passed
            // printf("====================in forward_many() ====================\n");
            // std::cout << "expert_idx: " << expert_idx << ", in_gpu_mask[expert_idx]: " << in_gpu_mask[expert_idx] << std::endl;
            if (in_gpu_mask[expert_idx]) return;
        }

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[KBackend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        // 由于
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[KBackend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);

    stride = QK_K; // 256
    nth = config_.hidden_size / stride; 
    // 计算 down_input @ down_proj
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        // printf("====================in forward_many() down before gpu mask====================\n");
        if(in_gpu_mask != nullptr) {
            // Use uint64_t for in_gpu_mask to match expert_ids
            // This assumes in_gpu_mask is a 1D array where each entry corresponds to an expert_id
            // If the expert_id is in GPU, skip the computation
            // Note: Ensure that in_gpu_mask is properly initialized and passed
            // std::cout << "expert_idx: " << expert_idx << ", in_gpu_mask[expert_idx]: " << in_gpu_mask[expert_idx] << std::endl;
            if (in_gpu_mask[expert_idx]) return;
        }
        
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[KBackend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif

        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);

    // 计算 output = down_output * weights
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights,
     const uint64_t* in_gpu_mask, 
     const void* input, void* output, int* batch_size_tensor, KBackend* backend) {
    // std::cout<< "MoE.forward() thread id:" <<std::this_thread::get_id() << std::endl;
    qlen = batch_size_tensor[0]; // qlen = batch_size * seq_len
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            // printf("====================moe.forward() before forward_one()====================\n");
            forward_one(k, 
                expert_ids + i * k, 
                weights + i * k, 
                in_gpu_mask, 
                (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
                (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
                 backend);
        }
        // printf("====================moe.forward() after forward_one()====================\n");
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);// 10 or qlen
    // printf("====================moe.forward() before forward_many()====================\n");
    forward_many(forward_len, k, expert_ids, weights, in_gpu_mask, input, output, backend);
    // printf("====================moe.forward() after forward_many()====================\n");

    batch_size_tensor[0] -= forward_len;
    forward(qlen - forward_len, 
        k, 
        expert_ids + forward_len * k, 
        weights + forward_len * k, 
        in_gpu_mask,
        (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
        (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
        batch_size_tensor, 
        backend);
}



int* MOE::replaceArray(const uint64_t* a, const uint64_t* b, int length) {
    // std::cout<< "MoE.replaceArray thread id:" <<std::this_thread::get_id() << std::endl;
    std::unordered_set<int> setA, setB;

    for (int i = 0; i < length; i++) {
        setA.insert(a[i]);
        setB.insert(b[i]);
    }

    // 求交集
    std::unordered_set<int> inter;
    for (int i = 0; i < length; i++) {
        if (setB.count(a[i])) {
            inter.insert(a[i]);
        }
    }

    // 提取 b 中的非交集元素
    int* b_extra = new int[length];
    int extraCount = 0;
    for (int i = 0; i < length; i++) {
        if (!inter.count(b[i])) {
            b_extra[extraCount++] = b[i];
        }
    }

    // 生成结果数组
    int* result = new int[length];
    int idx = 0;
    for (int i = 0; i < length; i++) {
        if (inter.count(a[i])) {
            result[i] = a[i];  // 保持交集元素不变
        } else {
            result[i] = b_extra[idx++]; // 替换
        }
    }

    delete[] b_extra; // 释放临时数组
    return result;
}

void MOE::prefetch(
        int update_policy,
        int prefetch_num,
        int cache_num,
        int pred_num,
        const uint64_t* expert_frequency,
        const uint64_t* pred_expert,
        uint64_t* cached_expert,
        uint64_t* up_slots,    // len = cache_num
        uint64_t* gate_slots,  // len = cache_num
        uint64_t* down_slots,  // len = cache_num
        int* cache_ready,
        uint64_t stream_ptr
    ){
        // 计算新的 cache 排列（保持交集原位，其他替换为 pred 里新的）
        std::unique_ptr<int[]> new_cache;

        if(update_policy == 0){
            new_cache.reset(get_new_cache_ids_v1(cached_expert, pred_expert, expert_frequency,
                                                cache_num, pred_num, prefetch_num));
        }else{
            new_cache.reset(get_new_cache_ids_v2(cached_expert, pred_expert, expert_frequency,
                                                cache_num, pred_num, prefetch_num));
        }
        
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

        int replace_num = 0;
        for(int i=0; i<cache_num; i++){
            // 完成cache的更新
            if(new_cache[i] != cached_expert[i]){
                if (replace_num >= prefetch_num) break;

                int new_expert_id = new_cache[i];
                load_ggml_expert_from_weights_c(new_expert_id, up_slots[i], gate_slots[i], down_slots[i], stream_ptr);
                cached_expert[i] = new_expert_id;
                replace_num++;
            }
        }

        // delete[] new_cache;
        cudaError_t err;
        err = cudaStreamSynchronize(stream);
        if(cache_ready) {
            *cache_ready = 1;
        }
        // printf("prefetch done, cache_ready=%d\n", cache_ready[0]);
    }

void MOE::load_ggml_expert_from_weights_c(
    int expert_id,
    uint64_t up_dst_ptr_val,
    uint64_t gate_dst_ptr_val,
    uint64_t down_dst_ptr_val,
    uint64_t stream_ptr
){

    // 将整数转回指针
    void* up_dst_ptr   = reinterpret_cast<void*>(up_dst_ptr_val);
    void* gate_dst_ptr = reinterpret_cast<void*>(gate_dst_ptr_val);
    void* down_dst_ptr = reinterpret_cast<void*>(down_dst_ptr_val);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // ==== 指针有效性检查 ====
    if (!up_proj_ || !gate_proj_ || !down_proj_ ||
        !up_proj_pinned || !gate_proj_pinned || !down_proj_pinned ||
        !up_dst_ptr || !gate_dst_ptr || !down_dst_ptr || !stream) 
    {
        fprintf(stderr, "[Warning] load_ggml_expert_from_weights_c: invalid pointer detected, skip expert_id=%d\n", expert_id);
        return;
    }
    // printf("in load_ggml_expert_from_weights_c!!!!!!!\n");
    // printf("\nexpert_id=%d\n", expert_id);
    size_t offset_gate = (size_t)expert_id
                * (size_t)config_.intermediate_size
                * (size_t)config_.hidden_size
                * (size_t)ggml_type_size(config_.up_type)
                / (size_t)ggml_blck_size(config_.up_type);
    void* gate_proj_ptr = (uint8_t*)gate_proj_ + offset_gate;

    size_t offset_up = (size_t)expert_id
                * (size_t)config_.intermediate_size
                * (size_t)config_.hidden_size
                * (size_t)ggml_type_size(config_.up_type)
                / (size_t)ggml_blck_size(config_.up_type);
    void* up_proj_ptr = (uint8_t*)up_proj_ + offset_up;

    size_t offset_down = (size_t)expert_id
                * (size_t)config_.hidden_size
                * (size_t)config_.intermediate_size
                * (size_t)ggml_type_size(config_.down_type)
                / (size_t)ggml_blck_size(config_.down_type);
    void* down_proj_ptr = (uint8_t*)down_proj_ + offset_down;


    // 可选：初始化数据，例如 memcpy 原始数据到 pinned 内存
    memcpy(up_proj_pinned, up_proj_ptr, up_nbytes);
    memcpy(gate_proj_pinned, gate_proj_ptr, gate_nbytes);
    memcpy(down_proj_pinned, down_proj_ptr, down_nbytes);

    // void* up_dst_ptr   = reinterpret_cast<void*>(up_dst_ptr_val);
    // void* gate_dst_ptr = reinterpret_cast<void*>(gate_dst_ptr_val);
    // void* down_dst_ptr = reinterpret_cast<void*>(down_dst_ptr_val);

    // 将整数转换回 CUDAStream
    // cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    cudaError_t err;
    err = cudaMemcpyAsync(up_dst_ptr,   up_proj_pinned,   up_nbytes,   cudaMemcpyHostToDevice,   stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy up failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(gate_dst_ptr, gate_proj_pinned, gate_nbytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy gate failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(down_dst_ptr, down_proj_pinned, down_nbytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy down failed: %s\n", cudaGetErrorString(err));
    }
}




// 非重复元素中的前prefetch_num个替换掉cache中频率最低的prefetch_num个，固定会替换prefetch_num个，如果足够的话。假设预测的前n个元素全部重复，则会顺延来产生替换
int* MOE::get_new_cache_ids_v1(const uint64_t* cached_expert, const uint64_t* pred_expert,
                          const uint64_t* expert_frequency, int cache_num,
                          int pred_num, int prefetch_num) {
    // 拷贝 cache
    std::vector<uint64_t> new_cache(cached_expert, cached_expert + cache_num);

    // pred 中的非重复元素
    std::unordered_set<uint64_t> cache_set(new_cache.begin(), new_cache.end());
    std::vector<uint64_t> pred_unique;
    for (int i = 0; i < pred_num; i++) {
        if (cache_set.find(pred_expert[i]) == cache_set.end()) {
            pred_unique.push_back(pred_expert[i]);
        }
    }

    // cache 中的非重复候选（可被替换）
    std::unordered_set<uint64_t> pred_set(pred_expert, pred_expert + pred_num);
    std::vector<std::pair<uint64_t,int>> cache_candidates;
    for (int i = 0; i < cache_num; i++) {
        if (pred_set.find(cached_expert[i]) == pred_set.end()) {
            cache_candidates.push_back({cached_expert[i], (int)expert_frequency[cached_expert[i]]});
        }
    }

    // 按频率升序排序
    std::sort(cache_candidates.begin(), cache_candidates.end(),
         [](auto &a, auto &b){ return a.second < b.second; });

    int replace_cnt = std::min(prefetch_num, (int)std::min(pred_unique.size(), cache_candidates.size()));

    for (int i = 0; i < replace_cnt; i++) {
        uint64_t old_id = cache_candidates[i].first;
        uint64_t new_id = pred_unique[i];
        // 找到 old_id 在 cache 中的位置并替换
        for (int j = 0; j < cache_num; j++) {
            if (new_cache[j] == old_id) {
                new_cache[j] = new_id;
                break;
            }
        }
    }

    int* result = new int[cache_num];
    for (int i = 0; i < cache_num; i++) result[i] = (int)new_cache[i];
    return result;
}

// 选取 pred 的前 prefetch_num 个，用其中的非重复元素替换掉 cache 中频率最低的 prefetch_num 个，可能会替换少于 prefetch_num 个，假设预测的前n个全部重复则不会发生替换
int* MOE::get_new_cache_ids_v2(const uint64_t* cached_expert, const uint64_t* pred_expert,
                          const uint64_t* expert_frequency, int cache_num,
                          int pred_num, int prefetch_num) {
    std::vector<uint64_t> new_cache(cached_expert, cached_expert + cache_num);

    // 取 pred 的前 prefetch_num
    std::vector<uint64_t> pred_topk;
    for (int i = 0; i < std::min(prefetch_num, pred_num); i++) {
        pred_topk.push_back(pred_expert[i]);
    }

    // cache 中的候选（非重复）
    std::unordered_set<uint64_t> pred_set(pred_topk.begin(), pred_topk.end());
    std::vector<std::pair<uint64_t,int>> cache_candidates;
    for (int i = 0; i < cache_num; i++) {
        if (pred_set.find(cached_expert[i]) == pred_set.end()) {
            cache_candidates.push_back({cached_expert[i], (int)expert_frequency[cached_expert[i]]});
        }
    }

    std::sort(cache_candidates.begin(), cache_candidates.end(),
         [](auto &a, auto &b){ return a.second < b.second; });

    int replace_cnt = std::min(prefetch_num, (int)cache_candidates.size());

    int pi = 0;
    for (int i = 0; i < replace_cnt && pi < (int)pred_topk.size(); i++) {
        uint64_t old_id = cache_candidates[i].first;
        uint64_t new_id = pred_topk[pi++];
        // 如果 new_id 已经在 cache 中，跳过
        if (std::find(new_cache.begin(), new_cache.end(), new_id) != new_cache.end()) {
            i--; // 保持替换数量一致，继续找下一个可替换的
            continue;
        }
        for (int j = 0; j < cache_num; j++) {
            if (new_cache[j] == old_id) {
                new_cache[j] = new_id;
                break;
            }
        }
    }

    int* result = new int[cache_num];
    for (int i = 0; i < cache_num; i++) result[i] = (int)new_cache[i];
    return result;
}