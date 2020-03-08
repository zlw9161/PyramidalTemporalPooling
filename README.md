### PyramidalTemporalPooling
Also named as Multi-Layer Temporal Pooling

Created by <a href="https://github.com/zlw9161">Liwen Zhang</a> and Jiqing Han from Speech Lab @ Harbin Institute of Technology, Harbin, China.

### Citation
If you find our work useful in your research, please cite:

        @article{Zhang:2020:dm-ptp,
          title={Pyramidal Temporal Pooling With Discriminative Mapping for Audio Classification},
          author={Liwen Zhang, Ziqiang Shi and Jiqing Han},
          Journal={IEEE Trans. Audio Speech and Language Proc.},
          year={2020}
          link={https://ieeexplore.ieee.org/document/8960462}
        }

### Abstract
An audio representation learning method with a hierarchical pyramid structure called pyramidal temporal pooling (PTP) which aims to capture the temporal information of an entire audio sample. By stacking a global temporal pooling layer on multiple local temporal pooling layers, the PTP can capture the high-level temporal dynamics of the input feature sequence in an unsuper-vised way.

### Contact
* Liwen Zhang (lwzhang9161@126.com and 15B903062@hit.edu.cn)

### Work Environment
* Matlab 2018a

### Dependency
* vlfeat-0.9.20
* liblinear-2.20
* libsvm-3.23
* matconvnet-1.0-beta25

### Dataset
* Audio Event Recognition dataset can be downloaded at:
https://bitbucket.org/naoya1/aenet_feat/src/master/
* Acoustic Scene Classification dataset can downloaded at:
http://dcase.community/challenge2018/task-acoustic-scene-classification#subtask-a

### Code Description
* ptp/defineNetwork.m<br />
This function is used for defining the structure of PTP network.
* ptp/passNetwork.m<br />
Feed forward pass of the defined PTP network.
* ptp/genRepresentation.m<br />
Generate audio representations with PTP network.
* ptp/getNonLinearity.m<br />
Non-linear feature mapping for the input feature before temporal encoding.
* ptp/liblinearsvr.m<br />
Linear support vector regression based temporal encoding.
* ptp/normalizeL2.m<br />
Feature normalization.
* ptp/rootExpandKernelMap.m<br />
PosNeg Expansion based Hellinger kernel.

### License
Our code is released under our License (see LICENSE file for details).

### Related Projects
* [TASLP 2017 paper - AENet: Learning Deep Audio Features for Video Analysis](http://arxiv.org/pdf/1701.00599) by Naoya Takahashi et al.
* [Interspeech 2018 paper - Unsupervised Temporal Feature Learning Based on Sparse Coding Embedded BoAW](https://github.com/zlw9161/VanillaTemporalPooling) by Liwen Zhang, Jiqing Han et al.

