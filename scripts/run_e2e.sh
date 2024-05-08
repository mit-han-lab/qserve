MODEL_PATH=./qserve_checkpoints/Llama-3-8B-Instruct-QServe # Please set the path accordingly

python qserve_e2e_generation.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path $MODEL_PATH \
  --group-size -1