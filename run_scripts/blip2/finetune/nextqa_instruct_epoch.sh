# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

#exp_name='blip2_nextqa_instruct_2e-5_e15'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
#--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
#--options run.output_dir=${result_dir}${exp_name} \
#run.init_lr=2e-5 \
#run.max_epoch=15

exp_name='blip2_nextqa_instruct_1e-5_e15'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=1e-5 \
run.max_epoch=15
