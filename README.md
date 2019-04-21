# MDNDC
In this paper, we propose a novel method called Multiple Deep Networks with scatter loss and Diversity Combination (MDNDC) for solving HFR problem. Firstly, to reduce intra-class and increase inter-class variations, the Scatter Loss (SL) is used as objective function which can bridge the modality gap while preserving the identity information. Secondly, we design a Multiple Deep Networks (MDN) structure for feature extraction, and propose a joint decision strategy called Diversity Combination (DC) to adaptively adjust weights of each deep network and make a joint classification decision. Finally, instead of using only one publicly available dataset, we make full use of multiple datasets to train the networks, which can further improve HFR performance.

# The trained models
## The model trained using SL loss
MS_Celeb_1M: We first pre-train the backbone network using MS_Celeb_1M dataset with softmax loss, and then fine-tune the network using CASIA NIR-VIS 2.0 dataset with SL loss. The Joint Bayesian is used as the classifier. The trained model can be found here (https://pan.baidu.com/s/1pfsUR6h3pk8r8AVCaQ1kpA  password: sruq). The network achieves rank-1 accuracy of 98.5 ± 0.3 and VR@FAR=0.1%(%) of 97.0 ± 0.5 on CASIA NIR-VIS 2.0, respectively.

VGGFace2: We first pre-train the backbone network using VGGFace2 dataset with softmax loss, and then fine-tune the network using CASIA NIR-VIS 2.0 dataset with SL loss. The Joint Bayesian is used as the classifier. The trained model can be found here (https://pan.baidu.com/s/1Vu_I9WZ9h6SnG28xGbec-w  password: aiwz). The network achieves rank-1 accuracy of 95.7 ± 0.5 and VR@FAR=0.1%(%) of 92.3 ± 0.8 on CASIA NIR-VIS 2.0, respectively.
 
CASIA_WebFace: We first pre-train the backbone network using CASIA_WebFace dataset with softmax loss, and then fine-tune the network using CASIA NIR-VIS 2.0 dataset with SL loss. The Joint Bayesian is used as the classifier. The trained model can be found here (https://pan.baidu.com/s/1PRvSxRAbMzngZdfUw93aUg  password: wmvd). The network achieves rank-1 accuracy of 92.3 ± 0.7 and VR@FAR=0.1%(%) of 88.4 ± 1.1 on CASIA NIR-VIS 2.0, respectively.

The three trained backbone networks and the MDNDC learned feature can be found here.

MS_Celeb_1M: https://pan.baidu.com/s/1pfsUR6h3pk8r8AVCaQ1kpA  password: sruq

VGGFace2: https://pan.baidu.com/s/1Vu_I9WZ9h6SnG28xGbec-w  password: aiwz

CASIA_WebFace: https://pan.baidu.com/s/1PRvSxRAbMzngZdfUw93aUg  password: wmvd

## DC model download
MDN network with DC fusion method: We first pre-train each backbone network using one of the MS_Celeb_1M, VGGFace2 or CASIA_WebFace datasets with softmax loss, and then fine-tune each network using CASIA NIR-VIS 2.0 or Oulu-CASIA NIR-VIS dataset with SL loss. Finally, we adopt the DC to fuse the three network. 

The MDNDC model achieves rank-1 accuracy of 98.9 ± 0.3 and VR@FAR=0.1%(%) of 97.6 ± 0.4 on CASIA NIR-VIS 2.0. Note that we only give results of one of the testing fold on CASIA NIR-VIS 2.0. The learned features for each network on this testing fold can be found in the project:

CASIA NIR-VIS 2.0:
NIR_VIS_DC_Joint_decision_CASIA_WebFace_single_feas.mat
NIR_VIS_DC_Joint_decision_MS_Celeb_1M_single_feas.mat
NIR_VIS_DC_Joint_decision_VGGFace2_single_feas.mat

One can test the MDNDC model on CASIA NIR-VIS 2.0 with MDNDC_CASIA_NIR_VIS_2_0_one_testingFold_DC.py. 

The MDNDC model achieves rank-1 accuracy of 99.8% and VR@FAR=0.1%(%) of 65.3% on Oulu-CASIA NIR-VIS. The learned features for each network on Oulu-CASIA NIR-VIS can be found in the project:

Oulu-CASIA NIR-VIS:
NIR_VIS_Oulu_DC_Joint_decision_CASIA_WebFace_single_feas.mat
NIR_VIS_Oulu_DC_Joint_decision_MS_Celeb_1M_single_feas.mat
NIR_VIS_Oulu_DC_Joint_decision_VGGFace2_single_feas.mat

One can test the MDNDC model on Oulu-CASIA NIR-VIS with MDNDC_Oulu_CASIA_NIR_VIS_DC.py. 

# Requirements
tensorflow 1.3.0 + 

cvxopt 1.2.0 

scipy 1.0.0 

scikit-learn 0.19.1 

# Backbone network
The inception-resnet-v1 network structure can be found here (https://github.com/davidsandberg/facenet). 

# Joint Bayesian classifier
The Joint Bayesian classifier can be found here (http://jiansun.org/papers/ECCV12_BayesianFace.pdf).

# Reference
[1] F. Schroff, D. Kalenichenko and J. Philbin, "Facenet: A unified embedding for face recognition and clustering," IEEE Conf. Computer Vision and Pattern Recognition, 2015, pp. 815-823.

[2] D. Chen, X. D. Cao, L. W. Wang, F. Wen and J. Sun, "Bayesian face revisited: a joint formulation,". Springer European Conference on Computer Vision, 2012, pp. 566-579.

[3] C. Szegedy, S. Ioffe, V. Vanhoucke and A. A. Alemi, "Inception-v4, inception-resnet and the impact of residual connections on learning," arXiv preprint arXiv:1602.07261, 2016.

