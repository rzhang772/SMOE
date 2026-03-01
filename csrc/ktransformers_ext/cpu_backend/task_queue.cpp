/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"
#include <iostream>
#include <thread>

TaskQueue::TaskQueue() {
    worker = std::thread(&TaskQueue::processTasks, this);// 一个工作线程负责按顺序执行队列中的任务即invoke(moe.forward())
    sync_flag.store(true, std::memory_order_seq_cst);
    exit_flag.store(false, std::memory_order_seq_cst);
}

TaskQueue::~TaskQueue() {
    {
        mutex.lock();
        exit_flag.store(true, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    // printf("in TaskQueue::enqueue!!!!!!!\n");
    {
        mutex.lock();
        tasks.push(task);
        sync_flag.store(false, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_one();
}

void TaskQueue::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst))
        ;
}

void TaskQueue::processTasks() {
    while (true) {
        std::function<void()> task;
        {
            mutex.lock();
            cv.wait(mutex, [this]() { return !tasks.empty() || exit_flag.load(std::memory_order_seq_cst); });
            if (exit_flag.load(std::memory_order_seq_cst) && tasks.empty()) {
                return;
            }
            task = tasks.front();
            tasks.pop();
            mutex.unlock();
        }
        // std::cout << "worker 线程 ID: " << worker.get_id() << std::endl;
        task();// moe.forward()
        {
            mutex.lock();
            if (tasks.empty()) {
                sync_flag.store(true, std::memory_order_seq_cst);
            }
            mutex.unlock();
        }
    }
}