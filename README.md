# DIFFNet

This repo is for **[Self-Supervised Monocular Depth Estimation with Internal Feature Fusion(arXiv)](https://arxiv.org/pdf/2110.09482.pdf), BMVC2021**

 A new backbone for self-supervised depth estimation.



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-monocular-depthestimation/monocular-depth-estimation-on-kitti-eigen-1)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1?p=self-supervised-monocular-depthestimation)
* Unlike other lead competitors, only test-time refinement (TTR) is applied on 1024x320 monocular model, no supervisory signals are involved with this rank.


If you think it is not a bad work, please consider citing it.

```
@article{zhou2021self,
  title={Self-Supervised Monocular DepthEstimation with Internal Feature Fusion},
    author={Zhou, Hang and Greenwood, David and Taylor, Sarah},
      journal={arXiv preprint arXiv:2110.09482},
        year={2021}
        }
```

## Comparing with others
![](images/table1.png)

## Evaluation on selected hard cases:
![](images/table2.png)

## Trained weights

- [diffnet_640x192](https://drive.google.com/file/d/1ZQPZWsIy_KyjV-Et6FSCOPM4iATjDPn-/view?usp=sharing)
- [diffnet_640x192_ms](https://drive.google.com/file/d/1_vh1F_cabTlEjBGXkHZOpAB1CMLmosxg/view?usp=sharing)
- [diffnet_1024x320](https://drive.google.com/file/d/1SuyBMS3ZLYuZwgyGSpmNrag7ESjRUC52/view?usp=sharing)
- [diffnet_1024x320_ttr](https://drive.google.com/file/d/1R0b0GYUxyZeaVCHQEELHevHoegwFi3qU/view?usp=sharing)

## Setting up before training and testing

- Data preparation: please refer to [monodepth2](https://github.com/nianticlabs/monodepth2)

## Training:

```
sh start2train.sh
```

* Note:

## Testing:

```
sh disp_evaluation.sh
```
## Infer a single depth map from a RGB:

```
sh test_sample.sh
```


#### Acknowledgement
 Thanks the authors for their works:
 - [monodepth2](https://github.com/nianticlabs/monodepth2)
 - [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)

