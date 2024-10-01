# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_ft_iv_v2_try2'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv_v2.yaml \
--options run.output_dir=${result_dir}${exp_name}

exp_name='blip2_nextqa_ft_iv_v2_try3'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv_v2_c1.yaml \
--options run.output_dir=${result_dir}${exp_name}

exp_name='blip2_nextqa_ft_iv_v2_try4'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv_v2_c2.yaml \
--options run.output_dir=${result_dir}${exp_name}