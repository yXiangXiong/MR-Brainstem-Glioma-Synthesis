# Pix2pixHD based Multi-task Generative Model
ISMRM Abstract: A multi-task generative network for simultaneous post-contrast MR image synthesis and tumor
segmentation: application to brainstem glioma

To reduce the exposure of Gadolinium-based Contrast Agents (GBCAs) in brainstem glioma detection and provide high-resolution contrast information,
we propose a novel multi-task generative network for contrast-enhanced T1-weight MR synthesis on brainstem glioma images. The proposed network
can simultaneously synthesize the high-resolution contrast-enhanced image and the segmentation mask of brainstem glioma lesions. <br><br>

## Image-to-image translation at 512x512 resolution

- {T1, T2, ASL}-to-{T1ce, tumor mask}
<p align='center'>
  <img src='imgs/BSG001_T1_1012.png' width='150'/>
  <img src='imgs/BSG001_T2_1012.png' width='150'/>
  <img src='imgs/BSG001_ASL_1012.png' width='150'/>
  <img src='imgs/BSG001_T1_1012_synthesized_image.jpg' width='150'/>
  <img src='imgs/BSG001_T1_1012_synthesized_mask.jpg' width='150'/>
</p>
<p align='center'>
  <img src='imgs/BSG043_T1_1007.png' width='150'/>
  <img src='imgs/BSG043_T2_1007.png' width='150'/>
  <img src='imgs/BSG043_ASL_1007.png' width='150'/>
  <img src='imgs/BSG043_T1_1007_synthesized_image.jpg' width='150'/>
  <img src='imgs/BSG043_T1_1007_synthesized_mask.jpg' width='150'/>
</p>

## Prerequisites
- Linux or Windows
- Python 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/yXiangXiong/Multi-task_Generative_Synthesis_Network
cd pix2pixHD_Multi-task_Learning
```


### Testing
- Test the model (`bash ./scripts/test_1024p.sh`):
```bash
#!./scripts/test.sh
python test.py --dataroot F:\xiongxiangyu\pix2pixHD_Mask_Data --name NC2C --label_nc 0 --input_nc 9 --output_nc 6 --resize_or_crop none --gpu_ids 0 --which_epoch 200 --no_instance --how_many 144
```
The test results will be saved to a html file here: `./results/NC2C/test_latest/index.html`.

More example scripts can be found in the `scripts` directory.

### Training
- Train a model at 512 x 512 resolution (`bash ./scripts/train_512p.sh`):
```bash
#!./scripts/train.sh
python train.py --dataroot F:\xiongxiangyu\pix2pixHD_Mask_Data --name NC2C --label_nc 0 --input_nc 9 --output_nc 6 --netG global --resize_or_crop none --gpu_ids 0 --batchSize 1 --no_instance
```
- To view training results, please checkout intermediate results in `./checkpoints/NC2C/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/NC2C/logs` by adding `--tf_log` to the training scripts.

## Citation
If you find this useful for your research, please use the following.

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
