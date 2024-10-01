# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_ft_tt_epoch'
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 --master_port 12345 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_ori_tt_v1_epoch.yaml \
--options run.output_dir=${result_dir}${exp_name}