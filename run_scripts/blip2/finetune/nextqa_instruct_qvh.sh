# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_instruct'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct_qvh.yaml \
--options run.output_dir=${result_dir}${exp_name}


exp_name='blip2_nextqa_instruct_3e-5'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct_qvh.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=3e-5