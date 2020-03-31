# Fast-MVSNet

PyTorch implementation of our CVPR 2020 paper:

[Fast-MVSNet: Sparse-to-Dense Multi-View Stereo With Learned Propagation and Gauss-Newton Refinement](https://arxiv.org/pdf/2003.13017.pdf)

Zehao Yu,
[Shenghua Gao](http://sist.shanghaitech.edu.cn/sist_en/2018/0820/c3846a31775/page.htm)

## How to use
```bash
git clone git@github.com:svip-lab/FastMVSNet.git
```
### Installation
 ```bash
pip install -r requirements.txt
```

### Training
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) from [MVSNet](https://github.com/YoYo000/MVSNet) and unzip it to ```data/dtu```.
* Train the network

    ```python fastmvsnet/train.py --cfg configs/dtu.yaml```
  
    You could change the batch size in the configuration file according to your own pc.

### Testing
* Download the [rectified images](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) from [DTU benchmark](http://roboimagedata.compute.dtu.dk/?page_id=36) and unzip it to ```data/dtu/Eval```.
    
* Test with the pretrained model

    ```python fastmvsnet/test.py --cfg configs/dtu.yaml TEST.WEIGHT outputs/pretrained.pth```

### Depth Fusion
We need to apply depth fusion ```tools/depthfusion.py``` to get the complete point cloud. Please refer to [MVSNet](https://github.com/YoYo000/MVSNet) for more details.

```bash
python tools/depthfusion.py -f dtu -n flow2
```

## Acknowledgements
Most of the code is borrowed from [PointMVSNet](https://github.com/callmeray/PointMVSNet). We thank Rui Chen for his great works and repos.

## Citation
Please cite our paper for any purpose of usage.
```
@inproceedings{Yu_2020_fastmvsnet,
  author    = {Zehao Yu and Shenghua Gao},
  title     = {Fast-MVSNet: Sparse-to-Dense Multi-View Stereo With Learned Propagation and Gauss-Newton Refinement},
  booktitle = {CVPR},
  year      = {2020}
}
```
