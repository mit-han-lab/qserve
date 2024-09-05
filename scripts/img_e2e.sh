MODEL_PATH=/home/shangy/dataset/models/VILA1.5/VILA1.5-7b/ # Please set the path accordingly
IMG_PATH=/home/shang/demo_images/climate_1.png,/home/shang/demo_images/animal_blocking.png

CUDA_VISIBLE_DEVICES=0 python qserve_e2e_generation_image.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w16a16kv8 \
  --quant-path $MODEL_PATH \
  --group-size -1 \
  --max-num-seqs 2 \
  --run-vlm --img-per-seq 1 \
  --img-files $IMG_PATH