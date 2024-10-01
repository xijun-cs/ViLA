# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/STAR/QA/"

exp_name='vlap_4f_test'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 12349 train.py \
--cfg-path lavis/projects/vila/train/star.yaml \
--options run.output_dir=${result_dir}${exp_name}