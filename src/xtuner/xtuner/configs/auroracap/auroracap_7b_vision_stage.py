import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, 
                          CLIPImageProcessor, CLIPVisionModel)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR

from torch.optim import AdamW
from xtuner.dataset import LLaVADataset, ConcatDataset, AuroraDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, aurora_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine import DatasetInfoHook, EvaluateChatHook
from xtuner.model import LLaVAModel, AuroraModel
from xtuner.model.aurora import AuroraEncoder
from xtuner.utils import PROMPT_TEMPLATE

    
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = 'lmsys/vicuna-7b-v1.5-16k'
visual_encoder_name_or_path = 'apple/DFN5B-CLIP-ViT-H-14-378'
pretrained_pth = 'model_path/projector'

prompt_template = PROMPT_TEMPLATE.vicuna
max_length = 4096
size = 378

batch_size = 1  # per_device
accumulative_counts = 24
lr = 1e-4
dataloader_num_workers = 0
max_epochs = 2
optim_type = AdamW
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03
visual_token_merge_ratio = 0.1
slowfast = False

# Save
save_steps = 100
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
    trust_remote_code=True,
    size=size,
    crop_size=size)

model = dict(
    type=AuroraModel,
    freeze_llm=True,
    freeze_visual_encoder=False,
    freeze_proj=False,
    slowfast=slowfast,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,)
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
data_root = 'data/dataset_name'
data_path = data_root + 'jsons/prompt_data.jsonl' # Change the data path to your exact path
image_folder = data_root
llava_dataset_1 = dict(
    type=AuroraDataset,
    data_path=data_path,
    # offline_processed_text_folder='', # Folder Path of the pre-processed dataset using 'bash scripts/preprocess_training_data.sh'
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=aurora_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

data_root = 'data/dataset_name'
data_path = data_root + 'jsons/prompt_data.jsonl' # Change the data path to your exact path
image_folder = data_root
llava_dataset_2 = dict(
    type=AuroraDataset,
    data_path=data_path,
    # offline_processed_text_folder='', # Folder Path of the pre-processed dataset using 'bash scripts/preprocess_training_data.sh'
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=aurora_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

data_root = 'data/dataset_name'
data_path = data_root + 'jsons/prompt_data.jsonl' # Change the data path to your exact path
image_folder = data_root
llava_dataset_3 = dict(
    type=AuroraDataset,
    data_path=data_path,
    # offline_processed_text_folder='', # Folder Path of the pre-processed dataset using 'bash scripts/preprocess_training_data.sh'
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=aurora_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)



train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        llava_dataset_1,  
        llava_dataset_2,
        llava_dataset_3])



train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Evaluate the generation performance during the training
evaluation_freq = 100
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']


# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)