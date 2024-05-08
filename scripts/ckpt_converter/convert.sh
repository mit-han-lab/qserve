MODEL_PATH=/dataset/models/llama2/llama-2-7b-hf/
QUANT_PATH=/home/shang/LM-serve/release/lmquant-dev/runs/llm/llama2/llama2-7b/w.4-x.8-y.4/w.zint4-x.sint8-y.zint4/w.gchn.fp16-x.gchn.fp16-y.g128.fp16/rot-smooth.xw.yx-w.static.kernel.orange/skip.y.[q]-krnl-rot-smth.xw.a0.b1.[AbsMax].[out+fc2].yx.a0p5.b0.[AbsMax].[qk]-w.range.[0p2.1.g80]-pileval.128x1024.[0-0]-240502.143030/

python checkpoint_converter.py --model-path $MODEL_PATH --quant-path $QUANT_PATH --group-size -1