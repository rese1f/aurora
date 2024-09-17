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

OUTPUT_DIR='work_dirs/llava_sft_internlm2_7b'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# 如果不是用 soft packing，则不需要数据缓存
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
  --llava work_dirs/llava_pretrain_internlm2_7b/20240724201849/hf-2180 \
  --tokenizer /mnt/hwfile/xtuner/huanghaian/model/internlm2-chat-7b \
  --chat-template 'internlm2' \
  --freeze-vit \
  --datasets data/llava_sft.json \
  --group-by-modality-length \
  --max-length 2048 \
  --num-workers 4 \
  --mirco-batch-size $MIRCO_BATCH_SIZE \
  --global-batch-size $((MIRCO_BATCH_SIZE*GPUS*ACCUMULATIVE_COUNTS)) \
  --lr 2e-5 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
