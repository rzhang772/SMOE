/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
 #ifndef CPUINFER_CPUINFER_H
 #define CPUINFER_CPUINFER_H
 
 #include <atomic>
 #include <condition_variable>
 #include <functional>
 #include <mutex>
 #include <queue>
 #include <thread>
 #include <vector>
 #include <stdexcept>
 #ifdef KTRANSFORMERS_USE_CUDA
 #include "vendors/cuda.h"
 #elif KTRANSFORMERS_USE_MUSA
 #include "vendors/musa.h"
 #elif KTRANSFORMERS_USE_ROCM
 #define __HIP_PLATFORM_AMD__
 #include "vendors/hip.h"
 #endif
 
 #include "backend.h"
 #include "task_queue.h"
 #include "./vendors/vendor.h"
 
 #include "llama.cpp/ggml-impl.h"
 
 class CPUInfer {
    public:
     CPUInfer(int thread_num) {
         backend_ = new KBackend(thread_num - 1);
         task_queue_ = new TaskQueue();
         prefetch_task_queue_ = new TaskQueue();
         for (int i = 0; i < (1 << 16); ++i) {
             ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
         }
     }
 
     ~CPUInfer() {
         delete backend_;
         delete task_queue_;
         delete prefetch_task_queue_;
     }
 
     template <typename Func, typename Obj, typename... Args>
     void enqueue(Func f, Obj* obj, Args... args) {
         task_queue_->enqueue([=]() {
             std::invoke(f, *obj, args..., backend_);
         });
     }
 
     // 接受参数为一个 std::pair<intptr_t, intptr_t>，其中第一个元素是函数指针，第二个元素是参数指针
     // 实际上python中moe类的forward绑定的不是C++实现的forward，而是在绑定时封装了一层绑定接口方法调用inner()在inner中执行入队cpuinfer.enqueue(moe.forward, args) -> task_queue.enqueue(invoke())
     // 实际绑定的forward接口会返回一个pair<inner, args>
     void submit(std::pair<intptr_t, intptr_t> params) {
         void (*func)(void*) = (void (*)(void*))params.first;
         void* args = (void*)params.second;// 包装的args
         *((CPUInfer**)args) = this;
         func(args);// inner()
     }
 
     void sync() {
         task_queue_->sync();
     }
 
     
     void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
        #if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM)
         void (*func)(void*) = (void (*)(void*))params.first;
         void* args = (void*)params.second;
         *((CPUInfer**)args) = this;
         cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
        #else
         throw std::runtime_error("submit_with_cuda_stream is not supported on this platforma");
        #endif
     }
 
     static void sync_(void* cpu_infer_ptr) {
         CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
         cpuinfer->sync();
     }
 
     void sync_with_cuda_stream(intptr_t user_cuda_stream) {
        #if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM)
         cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)this);
        #else
         throw std::runtime_error("sync_with_cuda_stream is not supported on this platforma");
        #endif
     }

     template <typename Func, typename Obj, typename... Args>
     void enqueue_prefetch(Func f, Obj* obj, Args... args) {
        // printf("in CPUInfer::enqueue_prefetch!!!!!!!\n");
         prefetch_task_queue_->enqueue([=]() {
             std::invoke(f, *obj, args...);
         });
     }

     void submit_prefetch(std::pair<intptr_t, intptr_t> params){
        // printf("in CPUInfer::submit_prefetch!!!!!!!\n");
        void (*func)(void*) = (void (*)(void*))params.first;// invoke(): 
        void* args = (void*)params.second;// 包装的args
        *((CPUInfer**)args) = this;
        func(args);
     }
 
    public:
     KBackend* backend_;
     TaskQueue* task_queue_;// task_queue中有一个线程负责监视队列，只要有任务就执行，具体执行时也就是moe.forward()中才调用backend中的线程池处理具体的计算任务

     TaskQueue* prefetch_task_queue_;
 };
 
 #endif