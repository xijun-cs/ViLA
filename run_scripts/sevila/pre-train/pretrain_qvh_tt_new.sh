result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='qvh_pretraining_tt_new'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 12346 train.py \
--cfg-path lavis/projects/sevila/train/qvh_tt.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.qvh.vis_processor.train.n_frms=4 \
datasets.qvh.vis_processor.eval.n_frms=75 \
run.batch_size_train=8 \
run.batch_size_eval=1 \
run.init_lr=3e-5 \
run.max_epoch=80 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
run.task='moment_retrieval'

