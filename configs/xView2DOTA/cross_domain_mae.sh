N_GPUS=2
BATCH_SIZE=4
DATA_ROOT=/input0
OUTPUT_DIR=../outputs/v1/xView2DOTA/0811_09_cross_domain_mae

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
--lr 2e-5 \
--lr_backbone 2e-6 \
--lr_linear_proj 2e-6 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode cross_domain_mae \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../0811_09_source_only/model_best.pth \

