# MindID
## Title: Mindid: Person identification from brain waves through attention-based recurrent neural network 

**PDF: [Ubicomp 2018](https://dl.acm.org/doi/10.1145/3264959), [arXiv](https://arxiv.org/abs/1711.06149)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Salil S Kanhere, Yunhao Liu, Tao Gu, Kaixuan Chen**

## Overview
This repository contains reproducible codes for the proposed MindID model.
Taking the advantages of EEG-based techniques for attack-resilient, we propose a biometric EEG-based identification approach, to overcome the limitations of traditional biometric identification methods. We analyzed the EEG data pattern characteristics and capture the Delta pattern which takes the most distinguishable features for user identification. Based on the pattern decomposition analysis, we report the structure of the proposed approach. In the first step of identification, the preprocessed EEG data is decomposed into Delta pattern. Then an attention-based RNN structure is employed to extract deep representations of Delta wave. At last, the deep representations are used to directly identify the user’ ID. Please check our paper for more details on the algorithm.

<p align="center">
<img src="https://raw.githubusercontent.com/xiangzhang1015/MindID/master/Flowchart%20of%20the%20proposed%20approach_MindID.PNG", width="700", align="center", title="Demonstration of the qualitative comparison. Our model can reconstruct all the shapes correctly which have the highest similarity with the ground truth.">
</p>
<div align='center'><b>Flowchart of the proposed MindID system.</b></div>
</p>


## Code
[MindID.py](https://github.com/xiangzhang1015/MindID/blob/master/MindID.py) is the Tensorflow based python code.


## Citing
If you find our work useful for your research, please consider citing this paper:

    @article{zhang2017mindid,
      title={MindID: Person identification from brain waves through attention-based recurrent neural network},
      author={Zhang, Xiang and Yao, Lina and Kanhere, Salil S and Liu, Yunhao and Gu, Tao and Chen, Kaixuan},
      journal={arXiv preprint arXiv:1711.06149},
      year={2017}
    }

## Datasets
- [EID-M.mat](https://github.com/xiangzhang1015/MindID/blob/master/EID-M.mat), [EID-S.mat](https://github.com/xiangzhang1015/MindID/blob/master/EID-S.mat) are two local datasets, please find more details description of the datasets in our paper.
- The public dataset EEG-S mentioned in the paper is a subset of the eegmmidb dataset (https://www.physionet.org/pn4/eegmmidb/). 
The generation of EEG-S is provided in the python code.


## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
