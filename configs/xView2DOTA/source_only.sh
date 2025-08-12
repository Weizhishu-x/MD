N_GPUS=2
BATCH_SIZE=4
DATA_ROOT=/input0
OUTPUT_DIR=../outputs/v1/xView2DOTA/0811_09_source_only

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26501 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone facebook/dinov2-large \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset xView \
--target_dataset DOTA \
--projector_scale 1.0 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--epoch 50 \
--epoch_lr_drop 40 \
--mode single_domain \
--output_dir ${OUTPUT_DIR}

