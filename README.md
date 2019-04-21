# MDNDC
We propose Scatter Loss objective function which can bridge the modality gap while preserving the identity information. Secondly, we design a Multiple Deep Networks structure for feature extraction, and propose a joint decision strategy called Diversity Combination to adaptively adjust weights of each deep network.

# The trained models
MS_Celeb_1M: We first pre-train the backbone network in MS_Celeb_1M dataset using softmax loss, and then fine-tune the network in CASIA NIR-VIS 2.0 dataset using SL loss, and the Joint Bayesian is used as the classifier. The trained model can be found here (https://pan.baidu.com/s/1pfsUR6h3pk8r8AVCaQ1kpA  password: sruq). The network achieves rank-1 accuracy of 98.5 ± 0.3 and VR@FAR=0.1%(%) of 97.0 ± 0.5 on CASIA NIR-VIS 2.0, respectively.

VGGFace2: We first pre-train the backbone network in VGGFace2 dataset using softmax loss, and then fine-tune the network in CASIA NIR-VIS 2.0 dataset using SL loss, and the Joint Bayesian is used as the classifier. The trained model can be found here (https://pan.baidu.com/s/1Vu_I9WZ9h6SnG28xGbec-w  password: aiwz). The network achieves rank-1 accuracy of 95.7 ± 0.5 and VR@FAR=0.1%(%) of 92.3 ± 0.8 on CASIA NIR-VIS 2.0, respectively.
 
CASIA_WebFace: We first pre-train the backbone network in CASIA_WebFace dataset using softmax loss, and then fine-tune the network in CASIA NIR-VIS 2.0 dataset using SL loss, and the Joint Bayesian is used as the classifier. The trained model can be found here (https://pan.baidu.com/s/1PRvSxRAbMzngZdfUw93aUg  password: wmvd). The network achieves rank-1 accuracy of 92.3 ± 0.7 and VR@FAR=0.1%(%) of 88.4 ± 1.1 on CASIA NIR-VIS 2.0, respectively.

MDN network with DC fusion method: We first pre-train each backbone network in MS_Celeb_1M, VGGFace2 or CASIA_WebFace dataset with softmax loss, and then fine-tune each network in CASIA NIR-VIS 2.0 dataset using SL loss. The Joint Bayesian is used as the classifier for each network. Finally, we adopt the DC to fuse the three network. The learned features for each network can be found here (https://pan.baidu.com/s/1b8hZf2vchHz2JvQ-JEbtnQ  password: az2c). The MDNDC model achieves rank-1 accuracy of 98.9 ± 0.3 and VR@FAR=0.1%(%) of 97.6 ± 0.4 on CASIA NIR-VIS 2.0, respectively. One can test the MDNDC model with Diversity_Combination.py. 

The three trained backbone networks and the MDNDC learned feature can be found here.

MS_Celeb_1M: https://pan.baidu.com/s/1pfsUR6h3pk8r8AVCaQ1kpA  password: sruq

VGGFace2: https://pan.baidu.com/s/1Vu_I9WZ9h6SnG28xGbec-w  password: aiwz

CASIA_WebFace: https://pan.baidu.com/s/1PRvSxRAbMzngZdfUw93aUg  password: wmvd

MDNDC learned features: https://pan.baidu.com/s/1b8hZf2vchHz2JvQ-JEbtnQ  password: az2c

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

