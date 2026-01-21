# README

This repository contains a PyTorch implementation of the paper **DIARY: Differentially Private Recovery with Adaptive Privacy Budgets in Federated Unlearning (WWW 2026).**

**Note: This repository will be updated in the next few days for readability and completeness. Please stay tuned!**

|                            DIARY                             |                  Privacy Budget Allocation                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./imgs/DIARY.png" alt="DIARY" style="zoom:25%;" /> | <img src="./imgs/Allocate_Budgets.png" alt="Allocate_Budgets" style="zoom:50%;" /> |

## 1. Setup
### Create a Conda Environment
```
# install the python
conda create -n DL python==3.8.0
conda activate DL
# install the pytorch and torchvision
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge
```
### Install Other Dependencies
```
pip install -r requirements.txt
```

## 2. Dataset Download and Divided
When you run the experiment, the dataset will be automatically downloaded and divided according to the configuration file. But it will only be done once for the same configuration file.


## 3. Run the Experiment
You can find some configuration files in folder `config` and run the following commands:
```bash
python main.py --config config/test/fmnist.yaml
```
```bash
python main_recover.py --config dba_flame_federaser
```
For more detailed parameters setting, you can check the configuration files.

## Acknowledgements

We would like to thank for Opacus repository.

