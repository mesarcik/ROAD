# `ROAD`: The Radio Observatory Anomaly Detector

A repository containing the implementation of the paper entitled [The ROAD to discovery: machine learning-driven anomaly detection in radio astronomy spectrograms](https://arxiv.org/abs/2307.01054)


## Installation 
Install conda environment by:
``` 
    conda create --name road python=3.9.7
``` 
Run conda environment by:
``` 
    conda activate road
```

Install the appropriate pytorch version:
``` 
    conda install pytorch torchvision torchaudio pytorch-cuda=<VERSION> -c pytorch -c nvidia
``` 

Install dependancies by running:
``` 
    pip install -r requirements
```

## Dataset  
You will need to download the [ROAD dataset](https://zenodo.org/record/8028045) and specify the its path using `-data_path` command line option.



## Replication of results in paper 
Run the following to replicate the results for the resnet34 used in the paper
```
    ./experiments/final_model.sh
```
or to run for all backbones 
```
    ./experiments/test.sh
```
Alternatively the [model weights](https://zenodo.org/record/8060501) can be downloaded and specified using the  `-model_name` and `-model_path` flags.


## Labelling with label-studio:
The labelling interface is based on [label-studio](https://labelstud.io/). To get the label server running for the LOFAR_AD project, run the following:
```
  label-studio start LOFAR_AD --sampling uniform &
```
and
```
./webserver /home/mmesarcik/data/LOFAR/compressed/LOFAR_AD/LOFAR_AD_v1/ *.png files 8081
````

## Licensing
Source code of ROAD is licensed under the MIT License.
