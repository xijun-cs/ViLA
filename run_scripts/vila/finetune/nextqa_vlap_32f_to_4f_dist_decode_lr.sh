# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/NextQA/QA/"

exp_name='nextqa_vlap_32f_to_4f_dist_dtemp_decode_bs8'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/vila/train/nextqa_vlap_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=5e-6

exp_name='nextqa_vlap_32f_to_4f_dist_dtemp_decode_1e-6_bs8'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/vila/train/nextqa_vlap_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=1e-6 \
run.warmup_steps=1000
