MODEL_PATH=/dataset/models/vilaqat/vilaqat-8b
QUANT_PATH=/home/shang/QAT/lmquant/projects/llm/runs/llm/vilaqat/vilaqat-8b/w.4-x.8-y.4/w.zint4-x.sint8-y.zint4/w.gchn.fp16-x.gchn.fp16-y.g128.fp16/w.static/skip.y.[q]-pileval.128x1024.[0-0]-240518.215424/

CUDA_VISIBLE_DEVICES=0 python checkpoint_converter.py --model-path $MODEL_PATH --quant-path $QUANT_PATH --group-size -1 --device cuda