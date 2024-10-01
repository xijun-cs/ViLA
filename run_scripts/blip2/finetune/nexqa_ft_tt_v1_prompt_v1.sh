# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_ft_tt_v1_prompt_v1'
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 --master_port 12346 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_ori_tt_v1_prompt_v1.yaml \
--options run.output_dir=${result_dir}${exp_name}


exp_name='blip2_nextqa_ft_ttt'
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 --master_port 12347 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_ori_ttt.yaml \
--options run.output_dir=${result_dir}${exp_name}
