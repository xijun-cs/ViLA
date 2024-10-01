# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_instruct'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
--options run.output_dir=${result_dir}${exp_name}
