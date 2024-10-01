# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/HOW2QA/QA/"

exp_name='vlap_4f_2e-5_ori'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/vila/train/how2qa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=3e-5 \
run.batch_size_train=16 \
run.warmup_steps=3000


exp_name='vlap_4f_3e-5_1000wu'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/vila/train/how2qa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=3e-5 \
run.warmup_steps=1000 \
run.accum_grad_iters=2

exp_name='vlap_4f_2e-5_1000wu_bs16'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
--cfg-path lavis/projects/vila/train/how2qa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=2e-5 \
run.batch_size_train=16 \
run.warmup_steps=1000 \
run.accum_grad_iters=2


