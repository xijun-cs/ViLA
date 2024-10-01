# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/HOW2QA/QA/"

exp_name='vlap_8t4f_dist_dtemp_decode_wp1200'
python -m torch.distributed.run --nproc_per_node=4 --master_port 12348 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.warmup_steps=1200 \
run.init_lr=2e-5

exp_name='vlap_8t4f_dist_dtemp_decode_wp1400'
python -m torch.distributed.run --nproc_per_node=4 --master_port 12348 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.warmup_steps=1400 \
run.init_lr=2e-5

exp_name='vlap_8t4f_dist_dtemp_decode_wp1600'
python -m torch.distributed.run --nproc_per_node=4 --master_port 12348 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.warmup_steps=1600 \
run.init_lr=2e-5

exp_name='vlap_8t4f_dist_dtemp_decode_wp2000'
python -m torch.distributed.run --nproc_per_node=4 --master_port 12348 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.warmup_steps=2000 \
run.init_lr=2e-5


