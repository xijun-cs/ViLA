# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_ft_tt_ttparam'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_ori_tt_ttparam.yaml \
--options run.output_dir=${result_dir}${exp_name}
