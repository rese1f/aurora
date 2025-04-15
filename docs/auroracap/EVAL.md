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

## VDC Evaluation
We have integrated the evaluation of VDC into the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalkit](https://github.com/open-compass/VLMEvalKit). Before launching the evaluation, you need to make sure that the GPU memory is sufficient to load both the test model and the evaluation model simultaneously.

If loading both the test model and the evaluation model simultaneously is inconvenient for you (e.g., due to limited GPU memory), you can separate the answer generation and score computation into two steps. Specifically, you should first replace the folder src/lmms-eval/lmms_eval/tasks/vdc with post_eval/vdc, and then run:
```
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=${SAVE_PATH} \
    --tasks ${AURORACAP_SUBSET} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${AURORACAP_SUBSET} \
    --output_path ${LOG_PATH}
```
This step replaces score computation and only generates the answers for later evaluation.

Second, use the json file generated in the first step to calculate the score of the corresponding subset. We use Llama3.1-8B as the evaluation model, and employ [SGLang](https://github.com/sgl-project/sglang) for acceleration. Before running the evaluation, we need to [install](https://docs.sglang.ai/start/install.html) and [launch](https://lmsys.org/blog/2024-07-25-sglang-llama3/) the SGLang server with 

```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct
```

Then run the following command to calculate the score of the corresponding subset (e.g. short caption).
```
python post_eval/process_vdc_result.py --raw_file path/to/input.json \
    --output_file path/to/output.jsonl \
    --tp_qa_path post_eval/short.jsonl
```
