BATCH_SIZE=4
DATA_ROOT=/input1
RESUME=/openbayes/home/wzs/q_MD/outputs/v1/HRRSD2SSDD/teaching_0720_16/model_best.pth
OUTPUT_DIR=/openbayes/home/wzs/q_MD/outputs/v1/HRRSD2SSDD/teaching_0720_16

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone facebook/dinov2-base \
--num_classes 2 \
--ca_nheads 8 \
--group_detr 10 \
--data_root ${DATA_ROOT} \
--source_dataset HRRSD \
--target_dataset SSDD \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--resume ${RESUME} \
--output_dir ${OUTPUT_DIR} \
--visualize
