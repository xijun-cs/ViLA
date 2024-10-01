# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/STAR/QA/"

exp_name='vlap_8f_lr'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/vila/train/star_8f.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.init_lr=3e-5 \
run.warmup_steps=1000 \
run.accum_grad_iters=2