# [ICML 2026] ConOrd

Chaewon Lee,
BeomJun Shim,
Kwang Pyo Choi,
and Chang-Su Kim

Official code for **"Contrastive Order Learning: A General Framework for Ordinal Regression"**

### Requirements
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 11.8
- cuDNN 8.7
- python 3.9
  
### Installation
Create conda environment:
```bash
    $ conda env create -f environment.yaml -n ConOrd
    $ conda activate ConOrd
```
Download repository:
```bash
    $ git clone https://github.com/cwlee00/ConOrd.git
```
Download weights:

ConOrd model [Google Drive](https://drive.google.com/drive/folders/1m4RPA0WlaRTro978DaZmO6TfJ-UHLerh)

### Evaluation
For evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/cwlee00/ConOrd/tree/main/config)

```
python test.py \
--checkpoint=./weights/ConOrd_models/ConOrd(clap).pth \
--dataset=clap
```
### Train
For training, please download the datasets, and then configure the path in [config.yml](https://github.com/cwlee00/ConOrd/tree/main/config)
```
python train.py \
--dataset=clap \
```

### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
    @InProceedings{Lee_2026_ConOrd_ICML,
    author    = {Lee, Chaewon and Shim, BeomJun and Choi, Kwang Pyo and Kim, Chang-Su},
    title     = {Contrastive Order Learning: A General Framework for Ordinal Regression},
    booktitle = {Forty-third International Conference on Machine Learning},
    year      = {2026}
    }
```

