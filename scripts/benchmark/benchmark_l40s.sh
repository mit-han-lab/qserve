# Benchmark script for QServe on A100. Batch size is decided by the device memory budget

# Download the model config files for benchmarking
MODEL_CONFIG_DIR_PATH=./QServe-benchmarks
if [ ! -d "$MODEL_CONFIG_DIR_PATH" ]; then
    git clone https://huggingface.co/datasets/mit-han-lab/QServe-benchmarks
fi

# Benchmark Llama-3-8B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-3-8B
GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=3200 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Llama-2-7B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-7B
GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=3200 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Mistral-7B
MODEL=$MODEL_CONFIG_DIR_PATH/Mistral-7B  
GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=3200 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Llama-2-13B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-13B  
GLOBAL_BATCH_SIZE=75 NUM_GPU_PAGE_BLOCKS=1875 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Llama-30B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-30B  
GLOBAL_BATCH_SIZE=32 NUM_GPU_PAGE_BLOCKS=800 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Yi-34B
MODEL=$MODEL_CONFIG_DIR_PATH/Yi-34B
GLOBAL_BATCH_SIZE=64 NUM_GPU_PAGE_BLOCKS=1600 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Llama-2-70B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-70B 
GLOBAL_BATCH_SIZE=24 NUM_GPU_PAGE_BLOCKS=600 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128

# Benchmark Qwen-1.5-72B
MODEL=$MODEL_CONFIG_DIR_PATH/Qwen-1.5-72B  
GLOBAL_BATCH_SIZE=4 NUM_GPU_PAGE_BLOCKS=100 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size 128
