# Multi-Modal Deep Convolutional Dictionary Learning for Image Denoising
Zhonggui Sun, Mingzhu Zhang, Huichao Sun, Jie Li, Tingting Liu, Xinbo Gao*, "Multi-Modal Deep Convolutional Dictionary Learning for Image Denoising," in Neurocomputing. (* Corresponding author)


The implementation of MMDCDicL is based on the [[DCDicL]] (https://github.com/cszn/KAIR).

## Requirement
- PyTorch 1.6+
- prettytable
- tqdm

## Testing
**Step 1**

- Download pretrained models from [[OneDrive]](https://1drv.ms/u/s!ApI9l49EgrUbjJ8cmYU4XBFUPutmag?e=AUEgnb) or [[BaiduPan]](https://pan.baidu.com/share/init?surl=vIqN2XiZ9UH8vcUpZPbXnw) (password: flfw).
- Unzip downloaded file and put the folders into ```./release/denoising```

**Step 2**

Configure ```options/test_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- path/pretrained_netG: path to the folder containing the pretrained models.


**Step 3**
```bash
python test_dcdicl.py
```



## Training
**Step 1**

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

**Step 2**


**Step 3**
```bash
python train_dcdicl.py
```

## Citation
```

```
