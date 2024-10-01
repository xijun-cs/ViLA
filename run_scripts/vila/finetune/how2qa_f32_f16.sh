# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/HOW2QA/QA/"

exp_name='vlap_32f'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 train.py \
--cfg-path lavis/projects/vila/train/how2qa_32f.yaml \
--options run.output_dir=${result_dir}${exp_name}

exp_name='vlap_16f'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 train.py \
--cfg-path lavis/projects/vila/train/how2qa_16f.yaml \
--options run.output_dir=${result_dir}${exp_name}