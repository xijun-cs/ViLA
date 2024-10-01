# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/NextQA/QA/"

exp_name='nextqa_vlap_32f_to_4f_dist_dtemp_decode'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/nextqa_vlap_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=5e-6

exp_name='nextqa_vlap_32f_to_4f_dist_dtemp_decode_1e-5'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/nextqa_vlap_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=1e-5 \
run.warmup_steps=1000

exp_name='nextqa_vlap_32f_to_4f_dist_dtemp_decode_1e-6'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/nextqa_vlap_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=1e-6


