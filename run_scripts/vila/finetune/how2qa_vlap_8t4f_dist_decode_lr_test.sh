# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/HOW2QA/QA/"

exp_name='vlap_8t4f_dist_dtemp_decode_wp1200_test'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12349 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.warmup_steps=1200 \
run.init_lr=2e-5


