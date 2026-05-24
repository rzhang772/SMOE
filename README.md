# SMOE

## 1. Configure Docker

Build and set up the Docker container according to the provided `Dockerfile`.

It is recommended to use VSCode to automatically complete the installation according to the configuration files in the `.devcontainer` directory.

## 2. Compile and Install

Run the installation script inside the container:
``` 
sh install.sh
```

## 3. Download the Model

Download the DeepSeek-V3 IQ4-XS model and prepare:

1. The original model directory (--model_path)
2. The corresponding GGUF file (--gguf_path)

Make sure both paths are correctly set before running inference.

## 4. Run Local Inference
```
python ./ktransformers/local_chat.py \
  --model_path your_model_path \
  --gguf_path your_gguf_path \
  --max_new_tokens 100 \
  --cpu_infer 8 \
  --prefetch_num 2 \
  --prefetch_method 0 \
  --prefetch_strategy 0 \
  --prefetch_start_layer 0 \
  --gpu_compute_max_num 8
```
### --prefetch_num:

Number of experts to prefetch each time.

### --prefetch_method:

0: token-wise

1: layer-wise

### --prefetch_strategy:

Replacement strategy:

0: Fixed number of replacements

1: Dynamic number of replacements

### --prefetch_start_layer:

Layer index from which prefetching starts.

### --gpu_compute_max_num:

Maximum number of experts handled by the GPU.

This parameter prevents GPU overload in cases where the expert hit rate is too high, which may otherwise result in GPU workload being significantly higher than CPU workload.

## Reminder: This repo gives the core token-wise prefetch and cache management implemention for Deepseek-V3 4bit.
This repository currently only includes the core implementations of token-wise prefetch and cache management. The remaining techniques will be integrated into this project in the future. The current codebase is experimental and relatively rough. If needed, please refine and modify the implementation according to your requirments and the methodology described in the paper. You are also welcome to contact us for further discussion.
Email:rui.zhang@connect.hkust-gz.edu.cn
