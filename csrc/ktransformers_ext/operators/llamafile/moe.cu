// #include "moe.h"
// #include <cuda_runtime.h>


// extern "C" void moe_alloc_pinned(
//     uint8_t** up_proj_pinned, size_t up_nbytes,
//     uint8_t** gate_proj_pinned, size_t gate_nbytes,
//     uint8_t** down_proj_pinned, size_t down_nbytes)
// {
//     cudaMallocHost(up_proj_pinned,   up_nbytes);
//     cudaMallocHost(gate_proj_pinned, gate_nbytes);
//     cudaMallocHost(down_proj_pinned, down_nbytes);
// }

// extern "C" void moe_free_pinned(
//     uint8_t* up_proj_pinned,
//     uint8_t* gate_proj_pinned,
//     uint8_t* down_proj_pinned)
// {
//     if (up_proj_pinned)   cudaFreeHost(up_proj_pinned);
//     if (gate_proj_pinned) cudaFreeHost(gate_proj_pinned);
//     if (down_proj_pinned) cudaFreeHost(down_proj_pinned);
// }

// void MOE::prefetch(
//         int prefetch_num,
//         int cache_num,
//         const void* input_tensor,
//         const uint64_t* expert_ids,
//         const uint64_t* pred_expert,
//         uint64_t* cached_expert,
//         uint64_t* up_slots,    // len = cache_num
//         uint64_t* gate_slots,  // len = cache_num
//         uint64_t* down_slots,  // len = cache_num
//         int* cache_ready,
//         uint64_t stream_ptr
//     ){
//         // printf("in prefetch!!!!!!!, cache_ready=%d\n", cache_ready[0]);
//         // printf("prefetch_num=%d, cache_num=%d\n", prefetch_num, cache_num);
//         // for(int i=0;i<cache_num;i++){
//         //     // std::cout<<"prefetch: pred_expert["<<i<<"]="<<pred_expert[i]<<", cached_expert["<<i<<"]="<<cached_expert[i]<<std::endl;
//         //     printf("prefetch: pred_expert[%d]=%ld, cached_expert[%d]=%d, expert_ids[%d]=%ld\n", i, pred_expert[i], i, cached_expert[i], i, expert_ids[i]);
//         // }

//     //     int dev;
//     //     cudaGetDevice(&dev);
//     //     printf("current device = %d\n", dev);
//     //     cudaDeviceProp prop;
//     //     cudaGetDeviceProperties(&prop, dev);

//     //     char uuid_str[64];
//     // snprintf(uuid_str, sizeof(uuid_str),
//     //          "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
//     //          (unsigned char)prop.uuid.bytes[0], (unsigned char)prop.uuid.bytes[1],
//     //          (unsigned char)prop.uuid.bytes[2], (unsigned char)prop.uuid.bytes[3],
//     //          (unsigned char)prop.uuid.bytes[4], (unsigned char)prop.uuid.bytes[5],
//     //          (unsigned char)prop.uuid.bytes[6], (unsigned char)prop.uuid.bytes[7],
//     //          (unsigned char)prop.uuid.bytes[8], (unsigned char)prop.uuid.bytes[9],
//     //          (unsigned char)prop.uuid.bytes[10], (unsigned char)prop.uuid.bytes[11],
//     //          (unsigned char)prop.uuid.bytes[12], (unsigned char)prop.uuid.bytes[13],
//     //          (unsigned char)prop.uuid.bytes[14], (unsigned char)prop.uuid.bytes[15]);

//     // printf("[C++] Current device: %d, %s, UUID=%s\n", dev, prop.name, uuid_str);
//         // 计算新的 cache 排列（保持交集原位，其他替换为 pred 里新的）
//         std::unique_ptr<int[]> new_cache(replaceArray(cached_expert, pred_expert, cache_num));
//         cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

//         // for(int i=0;i<cache_num;i++){
//         //     // std::cout<<"prefetch: new_cache["<<i<<"]="<<new_cache[i]<<std::endl;
//         //     printf("prefetch: new_cache[%d]=%d\n", i, new_cache[i]);
//         // }
//         // for(int i=0;i<cache_num;i++){
//         //     printf("up slots ptr val: up_slots[%d]=%lu\n", i, up_slots[i]);
//         // }

//         int replace_num = 0;
//         for(int i=0; i<cache_num; i++){
//             // 完成cache的更新
//             if(new_cache[i] != cached_expert[i]){
//                 if (replace_num >= prefetch_num) break;

//                 int new_expert_id = new_cache[i];
//                 load_ggml_expert_from_weights_c(new_expert_id, up_slots[i], gate_slots[i], down_slots[i], stream_ptr);
//                 cached_expert[i] = new_expert_id;
//                 replace_num++;
//             }
//         }

//         // delete[] new_cache;
//         if(cache_ready) {
//             *cache_ready = 1;
//         }
//         // printf("prefetch done, cache_ready=%d\n", cache_ready[0]);
//     }

// void MOE::load_ggml_expert_from_weights_c(
//     int expert_id,
//     uint64_t up_dst_ptr_val,
//     uint64_t gate_dst_ptr_val,
//     uint64_t down_dst_ptr_val,
//     uint64_t stream_ptr
// ){
//     // printf("in load_ggml_expert_from_weights_c!!!!!!!\n");
//     // printf("\nexpert_id=%d\n", expert_id);
//     size_t offset_gate = (size_t)expert_id
//                 * (size_t)config_.intermediate_size
//                 * (size_t)config_.hidden_size
//                 * (size_t)ggml_type_size(config_.up_type)
//                 / (size_t)ggml_blck_size(config_.up_type);
//     void* gate_proj_ptr = (uint8_t*)gate_proj_ + offset_gate;

//     size_t offset_up = (size_t)expert_id
//                 * (size_t)config_.intermediate_size
//                 * (size_t)config_.hidden_size
//                 * (size_t)ggml_type_size(config_.up_type)
//                 / (size_t)ggml_blck_size(config_.up_type);
//     void* up_proj_ptr = (uint8_t*)up_proj_ + offset_up;

//     size_t offset_down = (size_t)expert_id
//                 * (size_t)config_.hidden_size
//                 * (size_t)config_.intermediate_size
//                 * (size_t)ggml_type_size(config_.down_type)
//                 / (size_t)ggml_blck_size(config_.down_type);
//     void* down_proj_ptr = (uint8_t*)down_proj_ + offset_down;

//     // if(offset_gate < 0 || offset_gate > 256*2048*3808){
//     //     printf("Error: offset_gate out of range: %zu\n", offset_gate);
//     //     return;
//     // }
//     // if(offset_up < 0 || offset_up > 256*2048*3808){
//     //     printf("Error: offset_up out of range: %zu\n", offset_up);
//     //     return;
//     // }
//     // if(offset_down < 0 || offset_down > 256*7168*1088){
//     //     printf("Error: offset_down out of range: %zu\n", offset_down);
//     //     return;
//     // }

//     // 可选：初始化数据，例如 memcpy 原始数据到 pinned 内存
//     memcpy(up_proj_pinned, up_proj_ptr, up_nbytes);
//     memcpy(gate_proj_pinned, gate_proj_ptr, gate_nbytes);
//     memcpy(down_proj_pinned, down_proj_ptr, down_nbytes);

//     for(int i=0;i<16;i++){
//         // printf("", i, ((uint8_t*)up_proj_ptr)[i]);
//         printf("up_proj[%d]=%02x, up_proj_pinned[%d]=%02x\n", i, ((uint8_t*)up_proj_ptr)[i], i, ((uint8_t*)up_proj_pinned)[i]);
//     }

//     // printf("gate_nbytes=%d, up_nbytes=%d, down_nbytes=%d\n", gate_nbytes, up_nbytes, down_nbytes);

//     void* up_dst_ptr   = reinterpret_cast<void*>(up_dst_ptr_val);
//     void* gate_dst_ptr = reinterpret_cast<void*>(gate_dst_ptr_val);
//     void* down_dst_ptr = reinterpret_cast<void*>(down_dst_ptr_val);

//     // 将整数转换回 CUDAStream
//     cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
//     cudaError_t err;
//     err = cudaMemcpyAsync(up_dst_ptr,   up_proj_pinned,   up_nbytes,   cudaMemcpyHostToDevice,   stream);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Memcpy up failed: %s\n", cudaGetErrorString(err));
//     }

//     err = cudaMemcpyAsync(gate_dst_ptr, gate_proj_pinned, gate_nbytes, cudaMemcpyHostToDevice, stream);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Memcpy gate failed: %s\n", cudaGetErrorString(err));
//     }

//     err = cudaMemcpyAsync(down_dst_ptr, down_proj_pinned, down_nbytes, cudaMemcpyHostToDevice, stream);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Memcpy down failed: %s\n", cudaGetErrorString(err));
//     }
//     // printf("up_dst_ptr = %p\n", up_dst_ptr);

//     // cudaError_t err;
//     // err = cudaMemcpy(up_dst_ptr,   up_proj_pinned,   up_nbytes,   cudaMemcpyHostToDevice);
//     // if (err != cudaSuccess) {
//     //     fprintf(stderr, "Memcpy up failed: %s\n", cudaGetErrorString(err));
//     // }
//     // cudaMemcpy(gate_dst_ptr, gate_proj_pinned, gate_nbytes, cudaMemcpyHostToDevice);
//     // cudaMemcpy(down_dst_ptr, down_proj_pinned, down_nbytes, cudaMemcpyHostToDevice);

    
//     err = cudaStreamSynchronize(stream);
//         // if (err != cudaSuccess) {
//         //     fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
//         // } else {
//         //     printf("Stream synchronize success\n");
//         // }

        


// }