set -x

PARTITION=${PARTITION:-"llm_razor"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-8}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/llava_sft_internlm2_7b_pack'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

#   --resume \
# python -m debugpy --connect 10.140.0.31:5688 llava_pretrain.py \
MAX_LENGHT=2048
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun -p ${PARTITION} --time 1-00:00:00 \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u llava_train.py \
  --llava work_dirs/llava_pretrain_internlm2_7b_pack/20240904112552/hf-2114 \
  --tokenizer /mnt/hwfile/xtuner/huanghaian/model/internlm2-chat-7b \
  --chat-template 'internlm2' \
  --freeze-vit \
  --datasets data/llava_sft.json \
  --max-length $MAX_LENGHT \
  --pack-max-length $((MIRCO_BATCH_SIZE * MAX_LENGHT)) \
  --num-workers 4 \
  --group-by-length \
  --mirco-batch-size 1 \
  --global-batch-size $((GPUS*ACCUMULATIVE_COUNTS)) \
  --lr 2e-5 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  --dset-pack-level 'soft' \
  --dset-cache-dir /mnt/petrelfs/huanghaian/code/mm/xtuner/my_llava/llava_sft_cache \
  --dset-from-cache \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
