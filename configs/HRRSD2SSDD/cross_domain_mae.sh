N_GPUS=2
BATCH_SIZE=2
DATA_ROOT=/input1
OUTPUT_DIR=../outputs/v1/HRRSD2SSDD/0805_10_cross_domain_mae

# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
# --rdzv_endpoint localhost:26501 \
# --nproc_per_node=${N_GPUS} \
CUDA_VISIBLE_DEVICES=1 python main.py \
--backbone facebook/dinov2-base \
--num_classes 2 \
--data_root ${DATA_ROOT} \
--source_dataset HRRSD \
--target_dataset SSDD \
--projector_scale 1.0 \
--ca_nheads 8 \
--group_detr 10 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-5 \
--lr_backbone 2e-6 \
--lr_linear_proj 2e-6 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode cross_domain_mae \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../0805_10_source_only/model_best.pth \

