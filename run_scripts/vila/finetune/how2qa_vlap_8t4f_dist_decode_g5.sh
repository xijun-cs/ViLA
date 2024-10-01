# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/HOW2QA/QA/"

exp_name='vlap_8t4f_dist_dtemp_decode_2e-5'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=2e-5

exp_name='vlap_8t4f_dist_dtemp_decode_2e-5_gi8'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=2e-5 \
run.warmup_steps=1000 \
run.accum_grad_iters=8

exp_name='vlap_8t4f_dist_dtemp_decode_2e-5'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=2e-5 \
run.warmup_steps=1000 \
run.accum_grad_iters=4


exp_name='vlap_8t4f_dist_dtemp_decode_2e-5'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/how2qa_8f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=2e-5 \
run.warmup_steps=1000

