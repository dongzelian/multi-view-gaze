# Multi-View-Gaze

"Multi-view Multi-task Gaze Estimation with Deep Convolutional Neural Networks"

This paper is accepted by IEEE Transactions on Neural Networks and Learning Systems (TNNLS) 2018.

Our ShanghaiTechGaze dataset can be downloaded from [OneDrive](https://yien01-my.sharepoint.com/:u:/g/personal/doubility_z0_tn/EaDL9AkP5QdLgsOpmDw06K0BZqF0smTHMNOiH3ZMEk3WoA?e=VaKWeg).

About MPIIGaze dataset, please refer to the paper: "[Appearance-Based Gaze Estimation in the Wild](https://arxiv.org/pdf/1504.02863.pdf)". 

About UT Multiview dataset, please refer to the paper: "[Learning-by-Synthesis for Appearance-based 3D Gaze Estimation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909631)".

# Requirements
- python >= 3.6
- pytorch >= 0.4.1
- tensorboardX >= 1.4
- torchvision >= 0.2.1

# Usage
## Training our code

single-view on our ShanghaiTechGaze dataset 
```
python Train_Single_View_ST.py
```

multi-view on our ShanghaiTechGaze dataset 
```
python Train_Multi_View_ST.py
```

multi-view multi-task on the ShanghaiTechGaze and MPIIGaze dataset 
```
python Train_Multi_View_Multi_Task_ST_MPII.py
```
Sorry for the lack of pretrained model and partial code due to the migration of clustered environments.
Now the author is rewriting and reimplementing the code and experiments.

TODO: add multi-view multi-task implementation and inference code with our pretrained model.

# Citation

```
@article{lian2018tnnls,
    Author    = {Dongze Lian, Shenghua Gao, Lina Hu, Weixin Luo, Yanyu Xu, Lixin Duan, Jingyi Yu.},
    Title     = {Multi-view Multi-task Gaze Prediction with Deep Convolutional Neural Networks.},
    Journal   = {TNNLS},
    Year      = {2018}
    }
```