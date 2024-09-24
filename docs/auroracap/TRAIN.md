# Train
powered by [Xtuner](https://github.com/InternLM/xtuner), refer to the original [docs](../../src/xtuner/README.md) for more details.

## Environment preparation
Before launching training, you need to set up the environment to support Xtuner training..
```
cd src/xtuner
pip install -e '.[all]'
```

## Dataset preparation

Dataset prepare guides for original llava data can be found on [dataset_prepare.md](../../src/xtuner/docs/en/user_guides/dataset_prepare.md#dataset-prepare##others###llava_dataset). For the additional data from the other sources, we recommand convert into llava dataset format.

We recommand conduct pre-tokenization for all the training data for fast training launch. 

```
cd src/xtuner
python xtuner/tools/process_untokenized_llava_data.py \
    ${CONFIG_PATH} \
    --save-folder ${SAVE_PATH}
```

After pre-tokenization for the training data, you need to modify the config as follow:
```
data_root = 'data/dataset_name'
data_path = data_root + 'jsons/prompt_data.jsonl' # Change the data path to your exact path
image_folder = data_root
train_dataset = dict(
    type=AuroraDataset,
    # data_path=data_path,
    offline_processed_text_folder='', # Folder Path of the pre-processed dataset using 'bash scripts/preprocess_training_data.sh'
    image_folder=image_folder,
    # tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=aurora_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)
```

If the items in `data_path` is too much, such as over 1M, the CPU may overload during processing. Therefore, we recommend using the following code, which processes the data in multiple batches (we set the batch size to 100k in our configuration).
```
python xtuner/tools/process_untokenized_large_data.py \
    ${CONFIG_PATH} \
    --save-folder ${SAVE_FOLDER} \
    --input-file ${DATA_PATH} \
    --jsonl-output-folder ${TMP_FOLDER} \
```
`save-folder` means the folder to save the sharded pre-tokenization data, `jsonl-output-folder` stores the sharded jsonl files.

```
python scripts/merge_trunck.py \
    --data_folder ${SAVE_FOLDER} \
    --save_folder ${SAVE_PATH}
```
`data_folder` means the folder of the sharded pre-tokenization data, `save_folder` stores the merged pre-tokenization data and will output the number of the items and the first case.

## Training config

The training config example can be found in `src/xtuner/xtuner/configs/auroracap`. Here is an explanation of some important parameters.

`visual_token_merge_ratio` means the how many visual tokens being kept during training. For example, if it is set to be 0.1, then only 10% number of the visual tokens will be kept before sending to LLM.

`slowfast` means using slow-fast strategy proposed by [SlowFast-LLaVA](https://arxiv.org/abs/2407.15841), where we don't do token merging for the first frame.

## Launch training

To launch the training with a single GPU, use the following code:
```
cd src/xtuner
# On single GPU
python xtuner/tools/train.py \
    ${CONFIG_PATH} \
    --work-dir ${LOG_PATH} \
    --deepspeed deepspeed_zero2
# On multiple GPUs
# DIST 
python NPROC_PER_NODE=${GPU_NUM} xtuner/tools/train.py \
    ${CONFIG_PATH} \
    --work-dir ${LOG_PATH} \
    --deepspeed deepspeed_zero2
# SLURM
srun ${SRUN_ARGS} xtuner/tools/train.py \
    ${CONFIG_PATH} \
    --work-dir ${LOG_PATH} \
    --deepspeed deepspeed_zero2
```
- `--deepspeed` means using DeepSpeed to optimize the training. XTuner comes with several integrated strategies including ZeRO-1, ZeRO-2, and ZeRO-3. If you wish to disable this feature, simply remove this argument.

## Weight convert
After training, convert the saved `.pth` model (if using DeepSpeed, it will be a directory) to Hugging Face model, by 
```
python src/xtuner/xtuner/tools/model_converters/pth_to_hf.py \
    ${CONFIG_PATH}  \
    ${PTH_PATH} \
    ${SAVE_PATH} \
```
If the saved PTH model need to be the llava format, add `--save-format official`. If the saved PTH model need to be the safetensors format, add `--safe-serialization`.