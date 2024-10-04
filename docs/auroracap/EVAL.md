# Eval
powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Environment Prepare

For development, you can install the package by running the following command:
```
cd src/lmms-eval
pip install -e .
pip install flash-attn==2.3.6 --no-build-isolation
```

Since AuroraCap is based on the LLaVA model, you will need to install the required packages for LLaVA in order to test AuroraCap:
```
cd src/lmms-eval/LLaVA-NeXT
pip install -e .
```

## Weight convert
Before evaluation with lmms-eval, make sure the model is in Xtuner format.
```
python src/xtuner/xtuner/tools/model_converters/pth_to_hf.py \
    ${CONFIG_PATH}  \
    ${PTH_PATH} \
    ${SAVE_PATH} \
```


## Pretrained model
We also provide the pretrained AuroraCap model in Xtuner format

- 7B model for video captioning: `wchai/AuroraCap-7B-VID-xtuner`
- 7B model for image captioning: `wchai/AuroraCap-7B-IMG-xtuner`

replace the `pretrained` argument in the evaluation script with the model name above.

## Launch evaluation
We provide the example evaluation scripts:
```
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model auroracap \
    --model_args pretrained=${SAVE_PATH},token_merge_ratio=${ratio},slow_fast=${True/False} \
    --tasks ${BENCHMARK_NAME_1},${BENCHMARK_NAME_2} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${BENCHMARK_NAME_1, BENCHMARK_NAME_2} \
    --output_path ${LOG_PATH}
```
`num_processes` change it to perform multi node evaluation

`pretrained` checkpoint path

`token_merge_ratio` range from 0.01 to 1

`slow_fast` whether use slow-fast

`tasks` add yaml name under `src/lmms-eval/lmms_eval/tasks`, like `vatex_test`

`batch_size` we support only 1 for now.
