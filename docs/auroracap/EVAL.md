# Eval
powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Weight convert
Before evaluation with lmms-eval, make sure the model is a Hugging Face model.
```
python src/xtuner/xtuner/tools/model_converters/pth_to_hf.py \
    ${CONFIG_PATH}  \
    ${PTH_PATH} \
    ${SAVE_PATH} \
```

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