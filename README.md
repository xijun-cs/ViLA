# ViLA: Efficient Video-Language Alignment for Video Question Answering [ECCV2024]
In this work, we propose an efficient Video-Language Alignment (ViLA) network. Our ViLA model addresses both efficient frame sampling and effective cross-modal alignment in a unified way. In our ViLA network, we design a new learnable text-guided Frame-Prompter together with a new cross-modal distillation (QFormer-Distiller) module. Pre-trained large image-language models have shown promising results on problems such as visual question answering (VQA). However, how to efficiently and effectively sample video frames when adapting pre-trained large image-language model to video-language alignment is still the major challenge. Compared with prior work, our ViLA model demonstrates the capability of selecting key frames with critical contents, thus improving the video-language alignment accuracy while reducing the inference latency +3.3% on NExT-QA Temporal with 3.0X speed up).  Overall, our ViLA network outperforms the state-of-the-art methods on the video question-answering benchmarks: +4.6% on STAR Interaction, +2.2% on STAR average with 3.0X speed up, ours 2-frames out-perform SeViLA 4-frames on the VLEP dataset with 4.2X speed-up. 



# Code structure
```bash

# data & data preprocessing
./vila_data

# pretrained checkpoints
./vila_checkpoints


# ViLA code
./lavis/models/vila_models/


# running scripts for ViLA training
./run_scripts

```

# Setup

## Install Dependencies

1. (Optional) Creating conda environment

```bash
conda create -n vila python=3.8
conda activate vila
```

2. build from source

```bash
pip install -e .
```



# Dataset Preparation

We test our model on:
+ [NExT-QA](https://doc-doc.github.io/docs/nextqa.html)

+ [STAR](https://star.csail.mit.edu/)

+ [How2QA](https://value-benchmark.github.io/index.html)

+ [TVQA](https://tvqa.cs.unc.edu/)

+ [VLEP](https://value-benchmark.github.io/index.html)

+ [QVHighlights](https://github.com/jayleicn/moment_detr)

Please download original QA data and preprocess them via our [scripts](vila_data/).


# Training
We provide VLAP training script examples as follows.

And please change your data path.

## 1) Pre-training teacher
```bash
sh run_scripts/vila/finetune/star.sh
sh run_scripts/vila/finetune/star_8f.sh
sh run_scripts/vila/finetune/star_f32_f16.sh
```

## 2) prepare weight (change the model path first)

```bash
python re_weight.py
```

## 3) Training

```bash
sh run_scripts/vila/finetune/star_vila_32t4f_dist_decode.sh
```

## 3) Training with LoRA
Check ./lavis/models/vila_models/vila_lora.py

# Acknowledgments
We thank the developers of [SeViLA](https://github.com/Yui010206/SeViLA),  [LAVIS](https://github.com/salesforce/LAVIS), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [CLIP](https://github.com/openai/CLIP), [All-in-One](https://github.com/showlab/all-in-one), for their public code release.



## Citing Swin-MoE
```
@misc{wang2024vilaefficientvideolanguagealignment,
      title={ViLA: Efficient Video-Language Alignment for Video Question Answering},
      author={Xijun Wang and Junbang Liang and Chun-Kai Wang and Kenan Deng and Yu Lou and Ming Lin and Shan Yang},
      year={2024},
      eprint={2312.08367},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.08367},
}
```

# License

This project is licensed under the Apache-2.0 License.

