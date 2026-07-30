# DIARY [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18321979-blue)](https://doi.org/10.5281/zenodo.18321979)

This repository contains a PyTorch implementation of the paper **[DIARY: Differentially Private Recovery with Adaptive Privacy Budgets in Federated Unlearning](https://dl.acm.org/doi/abs/10.1145/3774904.3792423) (WWW 2026).**

**Note: This repository will be updated in the next few days for readability and completeness. Please stay tuned!**

|                            DIARY                             |                  Privacy Budget Allocation                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./imgs/DIARY.png" alt="DIARY" style="zoom:25%;" /> | <img src="./imgs/Allocate_Budgets.png" alt="Allocate_Budgets" style="zoom:50%;" /> |

**Our implementation extends the [Opacus](https://github.com/pytorch/opacus) library to support DIARY’s adaptive privacy budget allocation and privacy cost estimation, allowing each participant to set personalized, non-uniform budgets that align with their privacy preferences.**

## 1. Setup
### Create a Conda Environment
```
# install the python
conda create -n DIARY python==3.8.0
conda activate DIARY
# install the pytorch and torchvision
pip install pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```
### Install Other Dependencies
```
pip install -r requirements.txt
```

## 2. Dataset Download and Divided
When you run the experiment, the dataset will be automatically downloaded and divided according to the configuration file.


## 3. Run the Experiment
You can find some configuration files in folder `config` and run the following commands:
```bash
python main.py --config config/test/fmnist.yaml
```
For more detailed parameters setting, you can check the configuration files.

## 4. Learn More
- [Global privacy loss of various composition methods](./tutorials/global_privacy_loss_comparison.ipynb)
- [Simulation curvefitting](./tutorials/simulation_curvefitting.ipynb)

## 5. Citation
Please cite our paper if you use anything related in your work:
```
@inproceedings{10.1145/3774904.3792423,
    author = {Wang, Hengzhi and Dai, Lu and Zhang, Xianliang and Chen, Haoran and Hu, Juncheng and Yang, Kun},
    title = {DIARY: Differentially Private Recovery with Adaptive Privacy Budgets in Federated Unlearning},
    year = {2026},
    isbn = {9798400723070},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3774904.3792423},
    doi = {10.1145/3774904.3792423},
    booktitle = {Proceedings of the ACM Web Conference 2026},
    pages = {3042–3053},
    numpages = {12},
    keywords = {federated unlearning, differential privacy, model recovery},
    location = {United Arab Emirates},
    series = {WWW '26}
}
```

## Acknowledgements

We would like to thank for [Opacus](https://github.com/pytorch/opacus) library.

