# parameters/data path

result_dir="workspace"

exp_name='blip2_nextqa_infer'
ckpt='/prakhar/lamawaves/hub/checkpoints/blip2_pretrained_flant5xl.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path lavis/projects/blip2/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.finetuned=${ckpt} \
run.task='videoqa'