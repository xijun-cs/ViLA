# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_instruct_2e-5'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=2e-5 \
run.max_epoch=10

exp_name='blip2_nextqa_instruct_3e-5'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=3e-5 \
run.max_epoch=10


exp_name='blip2_nextqa_instruct_4e-5'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=4e-5 \
run.max_epoch=10


exp_name='blip2_nextqa_instruct_5e-5'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=5e-5 \
run.max_epoch=10
