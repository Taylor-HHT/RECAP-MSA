# RECAP: REcovering Coherent Affective Patterns Addressing Modality Missing in Multimodal Sentiment Analysis

### 🎉 Congrats! RECAP has been accepted by AAAI 2026 as an Oral Presentation

Pytorch implementation of the paper:
> **[Recovering Coherent Affective Patterns Addressing Modality Missing in Multimodal Sentiment Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/39349/43310)**

> This repository is a reorganized version of our codebase for public release. If you find any bugs or issues, please feel free to contact me.

## Content

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Training & Evaluation](#training--evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Overview

RECAP is a two-stage framework for multimodal sentiment analysis under modality missing settings:

1. Stage 1: modality completion
2. Stage 2: fusion prediction

![RECAP Framework](assets/RECAP_framework.png)


## Data Preparation

Please prepare the datasets in pickle format and place them under the `data/` directory in this repository:

- `data/mosi/unaligned_50.pkl`
- `data/mosei/unaligned_50.pkl`
- `data/sims/unaligned_39.pkl`

For downloading and processing MOSI, MOSEI, and SIMS, please see [MMSA](https://github.com/thuiar/MMSA).

If you store them elsewhere, update the paths in `configs/*.yaml`.

## Environment

The basic training environment for the results in the paper is Python 3.9.18, PyTorch 2.2.2, and CUDA 12.2 on NVIDIA RTX 4090 GPUs with 24GB memory.


## Training & Evaluation

### Step 1: Train the stage 1 completion model

Run stage 1 first to train the modality completion module:

```bash
python train.py --stage completion --config_file configs/train_mosi.yaml
```

After training finishes, a stage 1 checkpoint will be saved under `ckpt/<dataset_name>/` folder.


### Step 2: Train the stage 2 fusion model and evaluate

Then use the saved stage 1 checkpoint to train the fusion prediction stage:

```bash
python train.py \
  --stage fusion_prediction \
  --config_file configs/train_mosi.yaml \
  --stage1_ckpt ckpt/mosi/<your_stage1_checkpoint>.pth
```

If you prefer, you can also pass the stage 1 checkpoint suffix through `--time`, which will be expanded internally to `stage1_modules_<time>.pth`.

You can also obtain the validation and test results directly from the stage 2 training logs, and set `--missing_rate_eval_test` to evaluate under different missing rates.





## Acknowledgement

This repository is built upon the official codebase of [LNLN](https://github.com/Haoyu-ha/LNLN). We sincerely thank the authors for their open-source contribution.


## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{huang2026recovering,
  title={Recovering coherent affective patterns: Addressing modality missing in multimodal sentiment analysis},
  author={Huang, Huiting and Gong, Tieliang and He, Kai and Wen, Wen and Zhang, Weizhan and Feng, Mengling},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={26},
  pages={21957--21965},
  year={2026}
}
```
