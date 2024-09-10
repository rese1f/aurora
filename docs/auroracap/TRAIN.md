# Train
powered by [Xtuner](https://github.com/InternLM/xtuner), refer to the original [docs](../../src/xtuner/README.md) for more details.

## Dataset preparation

Dataset prepare guides for original llava data can be found on [dataset_prepare.md](../../src/xtuner/docs/en/user_guides/dataset_prepare.md#dataset-prepare##others###llava_dataset). For the additional data from the other sources, we recommand convert into llava dataset format.

We recommand conduct pre-tokenization for all the training data for fast training launch. 

```
cd src/xtuner
python xtuner/tools/process_untokenized_llava_data.py \
    ${CONFIG_PATH} \
    --save-folder ${SAVE_PATH}
```

## Training config

The training config example can be found in `src/xtuner/xtuner/configs/auroracap`. Here is an explanation of some important parameters.

`visual_token_merge_ratio` means the how many visual tokens being kept during training. For example, if it is set to be 0.1, then only 10% number of the visual tokens will be kept before sending to LLM.

`slowfast` means using slow-fast strategy proposed by [SlowFast-LLaVA](https://arxiv.org/abs/2407.15841), where we don't do token merging for the first frame.

## Launch training

To launch the training with a single GPU, use the following code:
```
cd src/xtuner
# On a single GPU
python xtuner/tools/train.py \
    ${CONFIG_PATH} \
    --work-dir ${LOG_PATH} \
    --deepspeed deepspeed_zero2
# On multiple GPUs
(DIST) python NPROC_PER_NODE=${GPU_NUM} xtuner/tools/train.py \
    ${CONFIG_PATH} \
    --work-dir ${LOG_PATH} \
    --deepspeed deepspeed_zero2
(SLURM)  srun ${SRUN_ARGS} xtuner/tools/train.py \
    ${CONFIG_PATH} \
    --work-dir ${LOG_PATH} \
    --deepspeed deepspeed_zero2
```

## Model format convert
