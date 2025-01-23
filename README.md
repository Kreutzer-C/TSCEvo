## Teacher-Student Co-Evolution for Source-Free Domain Adaptation with Vision-Language Model

------

This repo is the official implementation of our paper <>

### Getting Started

------

#### Environment Preparation

Please run the following command to install the required dependencies: 

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

#### Dataset Preparation

- Download PACS dataset from [here](https://drive.google.com/file/d/1TvEnu67YmMvdbBmvI7lygk19Q4M5PuK9/view?usp=drive_link)
- Download OfficeHome dataset from [here](https://drive.google.com/file/d/1LZ6O78jsARNb6zolvb6LyOGXt1wvmGg-/view?usp=drive_link)
- Download VLCS dataset from [here](https://drive.google.com/file/d/1oIKrAe892ICWnlbr6oDudD_1nszpYdO-/view?usp=drive_link)
- Download Terra Incognita dataset from [here](https://drive.google.com/file/d/1K1Nvinlpcmc8ftT_tR26H3AV767m6R0y/view?usp=drive_link)

Please place them in directory `./dataset` and extract them by running command `tar -xvzf file_name.tar.gz`

File structure like this is what we want:

```
├──dataset
    ├── PACS
    │   ├── art_painting
    │   ├── cartoon
    │   ├── photo
    │   └── sketch
    ├── OfficeHomeDataset_10072016
    │   ├── ...
    ├── VLCS
    │   ├── ...
    └── terra_incognita
        ├── ...
```

#### Launch Training

##### ERM source domain pre-train

Our first step is pre-training a source model by simple ERM method, to achieve this, running command:

```shell
python train_ERM.py --exp source_ERM
```

Please see `train_ERM.py` for more optional args setting. After pre-training, you can find the result at `./results/<--exp>/<--backbone>/<--dataset>` . 



Then, you need to register the pre-trained source model for further using in adaptation stage. Running command:

```shell
python register_ERM.py --exp source_ERM
```

Please note that you must set the value of `--exp` to an experiment name that has already completed pre-training via `train_ERM.py`



For better reproduction, we **highly recommand** you directly download the source model parameters that we have pre-trained. 

Download the model parameters from [here](https://drive.google.com/drive/folders/1sZuiP9gStOlCKsijeWa_oDRG4GYC_6_q?usp=drive_link) (only ResNet-50 version provided), then place them in directory `./pretrain/ERM`

Doing so would make it unnecessary to perform the above `train_ERM.py` and `register_ERM.py`

##### TSCEvo target domain adaptation

```shell
python train_TSCEvo.py\
-d Officehome -lr1 5e-4 -lr2 5e-5 --lamb3 10.0 --lamb4 0.5 --exp TSCEvo_default
```

```shell
python train_TSCEvo.py\
-d PACS -lr1 5e-4 -lr2 1e-4 --lamb3 10.0 --lamb4 0.5 --exp TSCEvo_default
```

```shell
python train_TSCEvo.py\
-d VLCS -lr1 5e-4 -lr2 1e-5 --lamb3 1.0 --lamb4 0.5 --exp TSCEvo_default
```

When dealing with Terra incognita dataset, you need to first download CLIP source fine-tuning LoRA params and initialize it. (denoted as CLIP(ft) in our paper). Download the LoRA parameters from [here](https://drive.google.com/drive/folders/1nxfj_jY2ZDdenPhp1BjPsc7r0U1yyovJ?usp=drive_link) (Only ViT-B/16 version provided), then place them in directory `./pretrain/LoRA`

```shell
python train_TSCEvo.py\
-d Terra -lr1 1e-3 -lr2 1e-6 --lamb3 10.0 --lamb4 0.5 --exp TSCEvo_default
```

Please see `train_TSCEvo.py` for more optional args setting. After training, you can find the result at `./results/<--exp>/<--dataset>/` . 



### Acknowledgments

The code of our work is partly built upon or draw inspiration from [RISE](https://github.com/WisconsinAIVision/RISE), [DIFO](https://github.com/tntek/source-free-domain-adaptation) and [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA) . Thanks for their contributions.