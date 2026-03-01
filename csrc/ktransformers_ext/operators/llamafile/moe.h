/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_H
#define CPUINFER_OPERATOR_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include <unordered_set>

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

struct MOEConfig {
    int expert_num;            // 256
    int routed_expert_num;// 8
    int hidden_size;// 7168
    int intermediate_size;// 2048
    int stride;// 64
    int group_min_len;// 10
    int group_max_len;// 1024
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;



    MOEConfig() {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class MOE {
   public:
    MOE(MOEConfig);
    ~MOE();
    void warm_up(KBackend* backend);
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, 
        const uint64_t* in_gpu_mask, 
        const void* input, 
        void* output, 
        KBackend* backend);
    void forward_many(int qlen, 
        int k, 
        const uint64_t* expert_ids, 
        const float* weights,
        const uint64_t* in_gpu_mask, 
        const void* input, 
        void* output, 
        KBackend* backend);
    void forward(int qlen, 
        int k, 
        const uint64_t* expert_ids, 
        const float* weights, 
        const uint64_t* in_gpu_mask, 
        const void* input, 
        void* output, 
        int* batch_size_tensor, 
        KBackend* backend);
    
    void prefetch(
        int update_policy, // 0: v1, 1: v2
        int prefetch_num,
        int cache_num,
        int pred_num,
        const uint64_t* expert_frequency, // expert频次统计
        const uint64_t* pred_expert,
        uint64_t* cached_expert,
        uint64_t* up_slots,    // len = cache_num
        uint64_t* gate_slots,  // len = cache_num
        uint64_t* down_slots,  // len = cache_num
        int* cache_ready,
        uint64_t stream
    );

    int* replaceArray(const uint64_t* a, const uint64_t* b, int length);
    int* get_new_cache_ids_v1(const uint64_t* cached_expert, const uint64_t* pred_expert,
                          const uint64_t* expert_frequency, int cache_num,
                          int pred_num, int prefetch_num);
    int* get_new_cache_ids_v2(const uint64_t* cached_expert, const uint64_t* pred_expert,
                          const uint64_t* expert_frequency, int cache_num,
                          int pred_num, int prefetch_num);
    
    void load_ggml_expert_from_weights_c(
        int expert_id,
        uint64_t up_dst_ptr_val,
        uint64_t gate_dst_ptr_val,
        uint64_t down_dst_ptr_val,
        uint64_t stream
    );
    

   private:
    MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    #ifdef USE_NUMA
    std::vector<void*> gate_proj_numa_;  // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> up_proj_numa_;    // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> down_proj_numa_;  // [numa_num, expert_num * hidden_size * intermediate_size ( /32 if quantized)]
    #endif

    float* s_input_fp32_;                      // [hidden_size]
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
    std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
    std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
    std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
    float* s_output_fp32_;                     // [hidden_size]

    std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
    std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    std::vector<uint8_t*> m_up_input_;    // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* m_local_gate_input_;         // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* m_local_up_input_;           // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* m_local_gate_output_;          // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_up_output_;            // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_intermediate_fp32_;    // [routed_expert_num * group_max_len * intermediate_size]
    uint8_t* m_local_down_input_;         // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* m_local_down_output_;          // [routed_expert_num * group_max_len * hidden_size]
    std::vector<float*> m_output_fp32_;   // [group_max_len, hidden_size]

    std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
    std::vector<int> m_local_num_;                       // [expert_num]
    std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
    std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
    std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
    std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
    std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
    std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
    std::vector<float*> m_local_down_output_ptr_;        // [expert_num]

    inline size_t up_bytes() const {
        // (hidden * inter) * (type_size / blck_size)
        return (size_t)config_.hidden_size * (size_t)config_.intermediate_size
             * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
    }
    inline size_t gate_bytes() const {
        return (size_t)config_.hidden_size * (size_t)config_.intermediate_size
             * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    }
    inline size_t down_bytes() const {
        // 注意 down 形状反过来 (inter * hidden)
        return (size_t)config_.intermediate_size * (size_t)config_.hidden_size
             * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    }

    size_t gate_nbytes = 0;
    size_t up_nbytes   = 0;
    size_t down_nbytes = 0;

    void* up_proj_pinned   = nullptr;
    void* gate_proj_pinned = nullptr;
    void* down_proj_pinned = nullptr;
};

#endif