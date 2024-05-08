# Benchmark script for QServe on A100. Batch size is decided by the device memory budget

# Download the model config files for benchmarking
MODEL_CONFIG_DIR_PATH=./QServe-benchmarks
if [ ! -d "$MODEL_CONFIG_DIR_PATH" ]; then
    git clone https://huggingface.co/datasets/mit-han-lab/QServe-benchmarks
fi

# Benchmark Llama-3-8B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-3-8B
GLOBAL_BATCH_SIZE=256 NUM_GPU_PAGE_BLOCKS=6400 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Llama-2-7B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-7B
GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=3200 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Mistral-7B
MODEL=$MODEL_CONFIG_DIR_PATH/Mistral-7B  
GLOBAL_BATCH_SIZE=256 NUM_GPU_PAGE_BLOCKS=6400 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Llama-2-13B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-13B  
GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=3200 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Llama-30B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-30B  
GLOBAL_BATCH_SIZE=64 NUM_GPU_PAGE_BLOCKS=1600 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Yi-34B
MODEL=$MODEL_CONFIG_DIR_PATH/Yi-34B
GLOBAL_BATCH_SIZE=196 NUM_GPU_PAGE_BLOCKS=4800 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Llama-2-70B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-70B 
GLOBAL_BATCH_SIZE=96 NUM_GPU_PAGE_BLOCKS=2400 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1

# Benchmark Qwen-1.5-72B
MODEL=$MODEL_CONFIG_DIR_PATH/Qwen-1.5-72B  
GLOBAL_BATCH_SIZE=32 NUM_GPU_PAGE_BLOCKS=780 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1
