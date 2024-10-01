# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"


exp_name='blip2_nextqa_instruct_32f_to_4f_dist_dtemp'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct_32f_to_4f_dist_temp.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=2e-6
