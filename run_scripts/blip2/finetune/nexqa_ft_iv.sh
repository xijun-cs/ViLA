# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_ft_iv_try3'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv.yaml \
--options run.output_dir=${result_dir}${exp_name}


exp_name='blip2_nextqa_ft_iv_try4'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv_c1.yaml \
--options run.output_dir=${result_dir}${exp_name}

exp_name='blip2_nextqa_ft_iv_try5'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv_c2.yaml \
--options run.output_dir=${result_dir}${exp_name}

exp_name='blip2_nextqa_ft_iv_try6'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_iv_c3.yaml \
--options run.output_dir=${result_dir}${exp_name}