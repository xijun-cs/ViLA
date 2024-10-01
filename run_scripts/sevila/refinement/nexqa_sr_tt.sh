# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='nextqa_sr_tt_2e-5'
ckpt='/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/nextqa_ft_sevila_tt_lr2e-5/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/sevila/train/nextqa_tt.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.nextqa.vis_processor.train.n_frms=4 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_train=8 \
run.batch_size_eval=4 \
run.init_lr=2e-5 \
run.max_epoch=10 \
run.warmup_steps=500 \
run.accum_grad_iters=2 \
model.task='train_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
