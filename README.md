# Echo from noise: synthetic ultrasound image generation using diffusion models for real image segmentation &middot; [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/your/your-project/blob/master/LICENSE) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://doi.org/10.48550/arxiv.2207.13424)

&nbsp;

<img src='./README_assets/pipeline.png'>  

## Papers

### [Echo from noise: synthetic ultrasound image generation using diffusion models for real image segmentation Paper](https://arxiv.org/abs/2207.00050)

[David Stojanovski](https://scholar.google.com/citations?user=6A_chPAAAAAJ&hl=en), [Uxio Hermida](https://scholar.google.com/citations?hl=en&user=6DkZyrXMyKEC), [Pablo Lamata](https://scholar.google.com/citations?hl=en&user=H98n1tsAAAAJ), [Arian Beqiri](https://scholar.google.com/citations?hl=en&user=osD0r24AAAAJ&view_op=list_works&sortby=pubdate), [Alberto Gomez](https://scholar.google.com/citations?hl=en&user=T4fP_swAAAAJ&view_op=list_works&sortby=pubdate)

### [Semantic Diffusion Model Paper](https://arxiv.org/abs/2207.00050)

[Weilun Wang](https://scholar.google.com/citations?hl=zh-CN&user=YfV4aCQAAAAJ), [Jianmin Bao](https://scholar.google.com/citations?hl=zh-CN&user=hjwvkYUAAAAJ), [Wengang Zhou](https://scholar.google.com/citations?hl=zh-CN&user=8s1JF8YAAAAJ), [Dongdong Chen](https://scholar.google.com/citations?hl=zh-CN&user=sYKpKqEAAAAJ), [Dong Chen](https://scholar.google.com/citations?hl=zh-CN&user=_fKSYOwAAAAJ), [Lu Yuan](https://scholar.google.com/citations?hl=zh-CN&user=k9TsUVsAAAAJ), [Houqiang Li](https://scholar.google.com/citations?hl=zh-CN&user=7sFMIKoAAAAJ),

## Abstract

We propose a novel pipeline for the generation of synthetic images via Denoising Diffusion Probabilistic Models (DDPMs)
guided by cardiac ultrasound semantic label maps. We show that these synthetic images can serve as a viable substitute
for real data in the training of deep-learning models for medical image analysis tasks such as image segmentation. To
demonstrate the effectiveness of this approach, we generated synthetic 2D echocardiography images and trained a neural
network for segmentation of the left ventricle and left atrium. The performance of the network trained on exclusively
synthetic images was evaluated on an unseen dataset of real images and yielded mean Dice scores of 88.5 $\pm 6.0$ , 92.3
$\pm 3.9$, 86.3 $\pm 10.7$ \% for left ventricular endocardial, epicardial and left atrial segmentation respectively.
This represents an increase of $9.09$, $3.7$ and $15.0$ \% in Dice scores compared to the previous state-of-the-art. The
proposed pipeline has the potential for application to a wide range of other tasks across various medical imaging
modalities.

## Example Results

&nbsp;

<img src='./README_assets/SDM_example_views.png'>  

&nbsp;

<img src='./README_assets/transforms.png'>  

&nbsp;

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Dataset Preparation

The data used and generated for the paper can be found as follows:

| Data                    | Download link                                                                                                                                   |
|:------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| CAMUS                   | [Image download](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8/folder/6373727d73e9f0047faa1bca) |
| SDM generated US images | [Image download](https://drive.google.com/file/d/1O8Avsvfc8rP9LIt5tkJxowMTpi1nYiik/view?usp=sharing)                                            |
| SDM pretrained weights  | [Checkpoint](https://drive.google.com/file/d/1iwpruJ5HMHdAA1tuNR8dHkcjGtxzSFV_/view?usp=sharing)                                                |

- Train the SDM model:

```bash
mpiexec -np 8 python3 ./image_train.py --datadir ./data/2ch_data --savedir ./output_2ch --batch_size_train 12 \
 --is_train True --save_interval 50000 --lr_anneal_steps 50000 --random_flip True --deterministic_train False \
 --img_size 256
```

- Inference the pretrained SDM model:

```bash
mpiexec -np 8 python3 ./image_sample.py --datadir ./data/2ch_data_augmented \
--resume_checkpoint ./output/ema_0.9999_050000_2ch.pt --results_dir ./results_2CH_ED --num_samples 1000 \
--is_train False --inference_on_train True
```

### Acknowledgements

Our code is developed based on [semantic-diffusion-model](https://github.com/WeilunWang/semantic-diffusion-model). 