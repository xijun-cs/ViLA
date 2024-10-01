# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/STAR/QA/"

exp_name='star_vlap_4f_to_4f_dist_dtemp_decode_1e-6'
ckpt='vila_checkpoints/star_vlap_blip_flanxl_trimmed_T_4f.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/star_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
model.finetuned=${ckpt} \
run.init_lr=1e-6

exp_name='star_vlap_4f_to_4f_dist_dtemp_decode_1e-6_bs16'
ckpt='vila_checkpoints/star_vlap_blip_flanxl_trimmed_T_4f.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12347 train.py \
--cfg-path lavis/projects/vila/train/star_32f_to_4f_dist_decode.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=1e-6 \
run.warmup_steps=1000 \
model.finetuned=${ckpt} \
run.accum_grad_iters=8
