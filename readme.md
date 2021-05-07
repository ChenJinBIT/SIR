# A Pytorch Implementation of [Sequential Instance Refinement for Cross-Domain Object Detection in Images](https://ieeexplore.ieee.org/abstract/document/9387548) (TIP 2021) 
<img src='./docs/Framework.pdf' width=900/>

## Introduction
Follow [faster-rcnn repository](https://github.com/jwyang/faster-rcnn.pytorch)
to setup the environment. 

We used Pytorch 0.4.1 and Python 3.6 for this project.
Follow [Strong-Weak Distribution Alignment for Adaptive Object Detection](https://github.com/VisionLearningGroup/DA_Detection)
to prepare data.

### An example of PASCAL VOC0712->Clipart
#### Train
* With S-agent and T-agent, ts and tt are thresholds of S-agent and T-agent.
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_instance_pixel_rl.py \
                    --dataset pascal_voc_0712 --dataset_t clipart --net res101 \
                    --use_tfb --ic \
                    --S --ts 0.5 --T --tt 0.5 \
                    --cuda
```

* When removing S-agent and T-agent,
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_instance_pixel_rl.py \
                    --dataset pascal_voc_0712 --dataset_t clipart --net res101 \
                    --use_tfb --ic \
                    --cuda
```

#### Test
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python test_net_instance_pixel_rl.py \
                    --dataset target_dataset --net res101 \
                    --cuda --ic \
                    --load_name path_to_model
```

### Citation
@article{chen2021sequential,
  title={Sequential Instance Refinement for Cross-Domain Object Detection in Images},
  author={Chen, Jin and Wu, Xinxiao and Duan, Lixin and Chen, Lin},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={3970--3984},
  year={2021},
  publisher={IEEE}
}