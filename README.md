# Multi-Modal Deep Convolutional Dictionary Learning for Image Denoising
Zhonggui Sun, Mingzhu Zhang, Huichao Sun, Jie Li, Tingting Liu, Xinbo Gao*, "Multi-Modal Deep Convolutional Dictionary Learning for Image Denoising," in Neurocomputing. (* Corresponding author)


The implementation of MMDCDicL is based on the [DCDicL](https://github.com/natezhenghy/DCDicL_denoising).

## Requirement
- PyTorch 1.6+
- prettytable
- tqdm
## Usage
### Testing
#### Modify Parameter
Configure ```options/test_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- path/pretrained_netG: path to the folder containing the pretrained models.


#### Runing
```bash
python test_dcdicl.py
```

### Training
#### Prepare training datasets
Prepare training/testing data. The folder structure should be similar to:

```
+-- data
|   +-- train
|       +-- Flash_non_Flash
|       +-- RGB_NIR
|   +-- test
|       +-- Flash_non_Flash
|       +-- RGB_NIR
```
#### Modify Parameter
Configure options/train_denoising.json. Important settings:

task: task name.
path/root: path to save the tasks.
data/train/sigma: range of noise levels.
netG/d_size: dictionary size.
netG/n_iter: number of iterations.
netG/nc_x: number of channels in NetX.
netG/nb: number of blocks in NetX.
test/visualize: true for saving the noisy input/predicted dictionaries.
If you want to reload a pretrained model, pay attention to following settings:
path/pretrained_netG: path to the folder containing the pretrained models.

#### Runing
```bash
python train_dcdicl.py
```
## Result
### Quantitative Results
![image](https://github.com/mingzhuzhang1/MMDCDicL/blob/main/Table1.png)
### Qualitative Results
<div align=center><img src="https://github.com/mingzhuzhang1/MMDCDicL/blob/main/Figure2.png"/></div>

## Acknowledgments
The authors would like to express their great thankfulness to the Associate Editor and the anonymous reviewers for
their valuable comments and constructive suggestions. At the same time, they would like to express their heartfelt thanks to the authors of the open source DCDicL. We recommend reading this article [DCDicL](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-final.pdf).
## Citation
