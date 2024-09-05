MODEL=/home/shang/QAT/qserve-dev-vlm/qserve_checkpoints/Llama-3-VILA1.5-8B-w4a8-per-chn # Please set the path accordingly

for batch in 16 32 64 128 256
do
    CUDA_VISIBLE_DEVICES=7 GLOBAL_BATCH_SIZE=$batch NUM_GPU_PAGE_BLOCKS=3600 RUN_VLM=1 IMG_PER_SEQ=1 IMG_ROTATION=0  python qserve_benchmark_image.py --model $MODEL --max-num-batched-tokens 262144 --max-num-seqs 256 --benchmarking --precision w4a8kv4 --group-size -1
done
