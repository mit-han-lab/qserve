MODEL_PATH=./qserve_checkpoints/vila1.5-8b-w4a8-per-channel # Please set the path accordingly

python qserve_e2e_generation.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path $MODEL_PATH \
  --group-size -1 \
  --max-num-seqs 1
