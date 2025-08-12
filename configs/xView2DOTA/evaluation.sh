BATCH_SIZE=16
DATA_ROOT=/input0
RESUME=./outputs/xView2DOTA/teaching_w_DT_enc/model_best.pth
OUTPUT_DIR=./outputs/xView2DOTA/teaching_w_DT_enc

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone facebook/dinov2-base \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset xView \
--target_dataset DOTA \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--resume ${RESUME} \
--output_dir ${OUTPUT_DIR} \
