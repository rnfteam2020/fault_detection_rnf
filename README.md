# Fault detection RNF project

## Description 
deadline - 15.12.2020

## Installation 
- install dependencies
```shell
pip install -r requirements.txt
```
- install [PyTorch](https://pytorch.org/get-started/locally/)

## Files structure

### core:
#### data
- generate "healthy and fault" data of 1DOF model mechanical oscillator

#### data_preprocessing
- input - data
- output - FFT, mean, max value ...
 
#### dataset
- generate torch dataset

#### nn_fit 
- Training functions for NN models

#### nn_models 
- NN models

#### visualization
- visualize results
- visualization functions

#### mat_loader
- wrapper for .mat format
 
### data:
#### datasets

## Sources
- [wiki](https://en.wikipedia.org/wiki/Fault_detection_and_isolation)
- [PyTorch: nn](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn)
- [link](https://www.sciencedirect.com/science/article/pii/S1876610218304831)
- [kaggle](https://www.kaggle.com/c/vsb-power-line-fault-detection/notebooks)
- [FDI](https://www.researchgate.net/publication/221412815_Fault_detection_methods_A_literature_survey/)
