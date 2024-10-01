# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

#exp_name='blip2_nextqa_instruct_t_160_2e-5'
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 train.py \
#--cfg-path lavis/projects/blip2/train/nextqa_instruct_t160.yaml \
#--options run.output_dir=${result_dir}${exp_name} \
#run.init_lr=2e-5
#
#exp_name='blip2_nextqa_instruct_tv2'
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 train.py \
#--cfg-path lavis/projects/blip2/train/nextqa_instruct_t_v2.yaml \
#--options run.output_dir=${result_dir}${exp_name} \
#run.init_lr=2e-5


exp_name='nextqa_ft_32f'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 train.py \
--cfg-path lavis/projects/sevila/train/nextqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=32 \
datasets.nextqa.vis_processor.train.n_frms=32 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_train=1 \
run.batch_size_eval=2 \
run.init_lr=1e-5 \
run.min_lr=1e-8 \
run.max_epoch=10 \
run.warmup_steps=500 \
run.accum_grad_iters=8 \
model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'