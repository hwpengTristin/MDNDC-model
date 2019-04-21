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

# Running Logs

## Running Logs on CASIA NIR-VIS 2.0
########################################################################
########################################################################
the backbone network is pre-trained in MS_Celeb_1M and fine-tuned in CASIA NIR-VIS 2.0

rank-1 and ROC results

accuracy of testing set is 0.985019%

fpr,tpr 1.3536342372000347e-06 0.6725193298969072./
fpr,tpr 1.0377862485200266e-05 0.7963917525773195./
fpr,tpr 0.00010016893355280256 0.9176868556701031
fpr,tpr 0.0010003357012908255 0.9710051546391752
fpr,tpr 0.010000198533021455 0.9924291237113402
fpr,tpr 0.10000018048456497 0.9988724226804123

########################################################################
########################################################################
the backbone network is pre-trained in VGGFace2 and fine-tuned in CASIA NIR-VIS 2.0

rank-1 and ROC results

accuracy of testing set is 0.957313%

fpr,tpr 1.3536342372000347e-06 0.3089561855670103

fpr,tpr 1.0377862485200266e-05 0.5346327319587629

fpr,tpr 0.00010016893355280256 0.8002577319587629

fpr,tpr 0.0010003357012908255 0.9231636597938144

fpr,tpr 0.010000198533021455 0.9838917525773195

fpr,tpr 0.10000018048456497 0.9980670103092784

########################################################################
########################################################################
the backbone network is pre-trained in CASIA_WebFace and fine-tuned in CASIA NIR-VIS 2.0

rank-1 and ROC results

accuracy of testing set is 0.922036%

fpr,tpr 1.3536342372000347e-06 0.25628221649484534

fpr,tpr 1.0377862485200266e-05 0.49065721649484534

fpr,tpr 0.00010016893355280256 0.7224548969072165

fpr,tpr 0.0010003357012908255 0.8841817010309279

fpr,tpr 0.010000198533021455 0.9658505154639175

fpr,tpr 0.10000018048456497 0.9959729381443299

########################################################################
########################################################################
The MDN network is pre-trained using three datasets and fine-tuned in CASIA NIR-VIS 2.0. Finally, we adopt DC to fuse the three networks

rank-1 and ROC results

accuracy of testing set is 0.989046%

fpr,tpr 1.3536342372000347e-06 0.7427512886597938

fpr,tpr 1.0377862485200266e-05 0.8266752577319587

fpr,tpr 0.00010016893355280256 0.9428157216494846

fpr,tpr 0.0010003357012908255 0.9764819587628866

fpr,tpr 0.010000198533021455 0.9961340206185567

fpr,tpr 0.10000018048456497 0.9995167525773195


## Running Logs on Oulu-CASIA NIR-VIS
########################################################################
########################################################################
the backbone network is pre-trained in MS_Celeb_1M and fine-tuned in Oulu-CASIA NIR-VIS
rank-1 and ROC results
960
accuracy of testing set is 0.987500%
fpr,tpr 1.1421783625730994e-06 0.2351345486111111
fpr,tpr 1.0279605263157895e-05 0.33676215277777777
fpr,tpr 0.00010051169590643275 0.4627821180555556
fpr,tpr 0.0010005482456140352 0.6358072916666667
fpr,tpr 0.010000913742690059 0.8706814236111111
fpr,tpr 0.10000114217836258 0.9970052083333333
########################################################################
########################################################################
the backbone network is pre-trained in VGGFace2 and fine-tuned in Oulu-CASIA NIR-VIS
rank-1 and ROC results
960
accuracy of testing set is 0.981250%
fpr,tpr 1.1421783625730994e-06 0.011393229166666666
fpr,tpr 1.0279605263157895e-05 0.018771701388888888
fpr,tpr 0.00010051169590643275 0.06545138888888889
fpr,tpr 0.0010005482456140352 0.3931206597222222
fpr,tpr 0.010000913742690059 0.7774956597222222
fpr,tpr 0.10000114217836258 0.9724392361111112
########################################################################
########################################################################
the backbone network is pre-trained in CASIA_WebFace and fine-tuned in Oulu-CASIA NIR-VIS
rank-1 and ROC results
960
accuracy of testing set is 0.959375%
fpr,tpr 1.1421783625730994e-06 0.0083984375
fpr,tpr 1.0279605263157895e-05 0.03357204861111111
fpr,tpr 0.00010051169590643275 0.0818142361111111
fpr,tpr 0.0010005482456140352 0.26156684027777777
fpr,tpr 0.010000913742690059 0.6899522569444444
fpr,tpr 0.10000114217836258 0.9709201388888888
########################################################################
########################################################################
The MDN network is pre-trained using three datasets and fine-tuned in Oulu-CASIA NIR-VIS. Finally, we adopt DC to fuse the three networks
rank-1 and ROC results
960
accuracy of testing set is 0.998958%
fpr,tpr 1.1421783625730994e-06 0.07973090277777778
fpr,tpr 1.0279605263157895e-05 0.19401041666666666
fpr,tpr 0.00010051169590643275 0.3381076388888889
fpr,tpr 0.0010005482456140352 0.6536675347222223
fpr,tpr 0.010000913742690059 0.8810763888888888
fpr,tpr 0.10000114217836258 0.998828125

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

