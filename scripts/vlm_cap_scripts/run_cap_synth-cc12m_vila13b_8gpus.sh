set -e
set -x

# read cli
job_id=${1:-0}

cd /home/shangy/qserve/release-vlm/qserve-dev
# source ~/.bashrc
# source activate qserve-v-release

dataset_path=/home/shangy/qserve/release-vlm/qserve-dev/cc3m-wds
info_path=/home/shangy/qserve/release-vlm/qserve-dev/cc3m-wds/cc_meta_train.json
model_path=/home/haotiant/workspace/checkpoints/VILA1.5-13b-qserve-w8a8

for gpu_id in {0..7}; do
  export CUDA_VISIBLE_DEVICES=$gpu_id
  python -W ignore qserve_vila_caption.py \
      --model $model_path \
      --quant-path $model_path \
      --precision w8a8kv8 \
      --ifb-mode \
      --group-size -1 \
      --max-num-seqs 32 \
      --run-vlm \
      --img-per-seq 1 \
      --omit-prompt \
      --max-new-tokens 300 \
      --data_path $dataset_path \
      --info_path $info_path \
      --gpu_id $gpu_id --job_id $job_id &
done

wait 

