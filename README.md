# NAD_Challenge - Ensemble method with voting post processing

## Introduction

This project is build for network anomaly detection (NAD) challenge held by ICASSP([Challenge Link](https://nad2021.nctu.edu.tw/index.html)). We propose an ensemble method with voting post processing thechnique to deteck real world firewall record data provided by ZYELL Communications Corp..

## Dataset

The dataset was provided by ZYELL Communication Corp., which is a real world firewall record data. The data contains 4 types of attack i.e. DDOS-smurf, Probing-IP sweep, Probing-Nmap sweep, and Probing-Port sweep.

## Getting Started

### Prerequisite and Environment Settings

The following packages should be installed in the first place.

**Ubuntu**

- python3, python3-pip
```shell
$ sudo apt-get update
$ sudo apt-get install -y python3 python3-pip
```
- virtualenv
```shell
$ sudo python3 -m pip install virtualenv
```

**Enviroment Setup**:
We use virtual environment to run this project, to setup envrionment you could use following script to setup environment.

```shell
$ virtualenv -p $(which python3) venv
$ . venv/bin/activate
$ python -m pip install -r requirements.txt
```

### run.sh

You could simpliy modify the paths of training and testing sets in ``run.sh``. Then, run ``run.sh`` directly to run this project. Note that you could also change `--pretrained` arguments to decide whether to use the pretrained models we provided.

```shell
$ sh run.sh
```

### Preprocessing

For data preprocessing, please use following script.

```shell
$ python preprocess.py --trn /path/to/training_data/training_data.csv --tst /path/to/testing_data/testing_data.csv --output_trn train.csv
```

### Training

For training model, please use following script.

```shell
$ python main.py --trn train.csv --tst_src /path/to/testing_data/testing_data.csv
```

### Prediction

For prediction, please use following script. Note that you should train your model first by running script above to create pretrained model.

```shell
$ python main.py --tst_src /path/to/testing_data/testing_data.csv --pretrained
```

