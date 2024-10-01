# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_instruct_8f'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct_8f.yaml \
--options run.output_dir=${result_dir}${exp_name}

exp_name='blip2_nextqa_instruct_16f'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct_16f.yaml \
--options run.output_dir=${result_dir}${exp_name}