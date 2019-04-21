from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def tf_kron(a, b):
    shape_a=np.shape(a)
    shape_b = np.shape(b)
    a_shape = [shape_a[0], shape_a[1]]
    b_shape = [shape_b[0], shape_b[1]]
    print(a_shape,b_shape)
    return tf.reshape(tf.reshape(a, [-1, a_shape[1], 1]) * tf.reshape(b, [-1, 1, b_shape[1]]),
                      [-1, a_shape[1] * b_shape[1]])

def tf_kron_vector(a, b):
    # shape_a = a.get_shape().as_list()
    # shape_b = a.get_shape().as_list()
    shape_a=np.shape(a)
    shape_b = np.shape(b)
    a_shape = [shape_a[0], shape_a[1]]
    b_shape = [shape_b[0], shape_b[1]]

    return tf.reshape(tf.reshape(a, [a_shape[0], 1, a_shape[1], 1]) * tf.reshape(b, [1, b_shape[0], 1, b_shape[1]]),
                      [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])

def main():
  

    import scipy.io as sio


    ########################################################################
    dataset='MS_Celeb_1M_single'
    data = sio.loadmat('./' + 'NIR_VIS_Oulu_DC_Joint_decision_' + dataset + '_feas.mat')
    rank_val_s_vector_train1= data['rank_val_s_vector_train'] # s_i^1
    probe_label1=data['probe_label'] # the probe label
    gallery_label1=data['gallery_label'] # the gallery label
    Probe_dis_List_matric1 = data['Probe_dis_List_matric'] # d_{ij}^1: each element denotes the distance between the ith probe and the jth gallery on the 1th backbone network (Node: the distance is calculated by Joint Bayesian)



    ########################################################################
    dataset='VGGFace2_single'
    data = sio.loadmat('./' + 'NIR_VIS_Oulu_DC_Joint_decision_' + dataset + '_feas.mat')
    rank_val_s_vector_train2= data['rank_val_s_vector_train'] # s_i^2
    probe_label2=data['probe_label'] # the probe label
    gallery_label2=data['gallery_label'] # the gallery label
    Probe_dis_List_matric2 = data['Probe_dis_List_matric'] # d_{ij}^2: each element denotes the distance between the ith probe and the jth gallery on the 2th backbone network (Node: the distance is calculated by Joint Bayesian)



    ########################################################################
    dataset='CASIA_WebFace_single'
    data = sio.loadmat('./' + 'NIR_VIS_Oulu_DC_Joint_decision_' + dataset + '_feas.mat')
    rank_val_s_vector_train3= data['rank_val_s_vector_train'] # s_i^3
    probe_label3=data['probe_label'] # the probe label
    gallery_label3=data['gallery_label'] # the gallery label
    Probe_dis_List_matric3 = data['Probe_dis_List_matric'] # d_{ij}^3: each element denotes the distance between the ith probe and the jth gallery on the 3th backbone network (Node: the distance is calculated by Joint Bayesian)



    ########################################################################
    # obtain three weight

    subset_ratio=1.0
    train_num=int(len(rank_val_s_vector_train1[0])*subset_ratio)
    rank_val_s_vector_train1=rank_val_s_vector_train1[0][0:train_num]
    rank_val_s_vector_train2 = rank_val_s_vector_train2[0][0:train_num]
    rank_val_s_vector_train3 = rank_val_s_vector_train3[0][0:train_num]

    vec_s =np.empty((3,1))


    net1_weight=0
    net2_weight=0
    net3_weight=0

    from cvxopt import solvers, matrix

    for i in range(len(rank_val_s_vector_train1)):
        vec_s[0, 0]=rank_val_s_vector_train1[i]

        vec_s[1, 0] = rank_val_s_vector_train2[i]

        vec_s[2, 0] = rank_val_s_vector_train3[i]

        vec_s=np.mat(vec_s)


        P=matrix((vec_s*vec_s.T).T)

        q=matrix([0.0,0.0,0.0])

        G=matrix([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]])
        h=matrix([0.0,0.0,0.0])

        A=matrix([[1.0],[1.0],[1.0]])

        b=matrix([1.0])

        sol=solvers.qp(P,q,G,h,A,b)
        # print(vec_s)
        # print(sol['x'])
        # print(sol['x'][0],sol['x'][1],sol['x'][2])
        net1_weight += sol['x'][0]
        net2_weight += sol['x'][1]
        net3_weight += sol['x'][2]
    ########################################################################
    print('########################################################################')
    print('########################################################################')
    print('DC fusion method to obtain weight vector')
    print('net1_weight,net2_weight,net3_weight',net1_weight,net2_weight,net3_weight)


    net1=net1_weight/len(rank_val_s_vector_train1) # \delta_1
    net2=net2_weight/len(rank_val_s_vector_train1) # \delta_2
    net3=net3_weight/len(rank_val_s_vector_train1) # \delta_3

    scale=3 # \tau
    net1=math.pow(net1,scale)
    net2=math.pow(net2,scale)
    net3=math.pow(net3,scale)

    net_sum=(net1+net2+net3)
    net1 = net1 / net_sum # \eta_1
    net2 = net2 / net_sum # \eta_2
    net3 = net3 / net_sum # \eta_3

    print(net1,net2,net3,net1+net2+net3)




    ########################################################################
    # MS_Celeb_1M: rank-1 and ROC results
    print('########################################################################')
    print('########################################################################')
    print('the backbone network is pre-trained in MS_Celeb_1M and fine-tuned in Oulu-CASIA NIR-VIS')
    print('rank-1 and ROC results')
    print(len(Probe_dis_List_matric1[0]))

    ture_samples=0
    probe_dis_List=[]
    probe_pairLabel_list=[]


    count=0
    for idx in range(Probe_dis_List_matric1.shape[0]):


        net1_dis_vec=Probe_dis_List_matric1[idx,:]*net1
        net2_dis_vec = Probe_dis_List_matric2[idx, :] * net2
        net3_dis_vec = Probe_dis_List_matric3[idx, :] * net3
        net_dis_vec=net1_dis_vec+net2_dis_vec+net3_dis_vec

        minVals = 100000
        minIndex=-1
        count += 1
        # for index,val in enumerate(net_vec):
        ###according to dist
        for index, val in enumerate(net1_dis_vec):

            # probe_dis_List.append(-val)
            probe_dis_List.append(-net1_dis_vec[index])


            diff=val
            if diff < minVals:
                minIndex = index
                minVals=diff

            if gallery_label1[0,index] == probe_label1[0,idx]:
                probe_pairLabel_list.append(1)
            else:
                probe_pairLabel_list.append(0)

        if gallery_label1[0,minIndex] == probe_label1[0,idx]:

            ture_samples += 1

        if idx%1000==0 and idx>0:
            print("testing process:" + str(idx)+'/'+str(Probe_dis_List_matric1.shape[0])+'  |  '\
                  +"current acc: " +str(ture_samples/count) )

    acc = (float(ture_samples) / Probe_dis_List_matric1.shape[0])
    print('accuracy of testing set is %f%%' % acc)  # RANK-1 accuracy

    # TODO ROC
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc  ### roc, auc
    import matplotlib.pyplot as plt
    fpr, tpr, threshold = roc_curve(probe_pairLabel_list, probe_dis_List,drop_intermediate=False)  ### fpr, tpr
    roc_auc = auc(fpr, tpr)  ### auc

    for idx, val in enumerate(fpr):
        if val > 0.000001 and val < 0.0000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.00001 and val < 0.000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.0001 and val < 0.00019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.001 and val < 0.0019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.01 and val < 0.019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.1 and val < 0.19:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break


    ########################################################################
    # VGGFace2: rank-1 and ROC results
    print('########################################################################')
    print('########################################################################')
    print('the backbone network is pre-trained in VGGFace2 and fine-tuned in Oulu-CASIA NIR-VIS')
    print('rank-1 and ROC results')
    print(len(Probe_dis_List_matric1[0]))

    ture_samples=0
    probe_dis_List=[]
    probe_pairLabel_list=[]


    count=0
    for idx in range(Probe_dis_List_matric1.shape[0]):


        net1_dis_vec=Probe_dis_List_matric1[idx,:]*net1
        net2_dis_vec = Probe_dis_List_matric2[idx, :] * net2
        net3_dis_vec = Probe_dis_List_matric3[idx, :] * net3
        net_dis_vec=net1_dis_vec+net2_dis_vec+net3_dis_vec

        minVals = 100000
        minIndex=-1
        count += 1
        # for index,val in enumerate(net_vec):
        ###according to dist
        for index, val in enumerate(net2_dis_vec):

            # probe_dis_List.append(-val)
            probe_dis_List.append(-net2_dis_vec[index])


            diff=val
            if diff < minVals:
                minIndex = index
                minVals=diff

            if gallery_label1[0,index] == probe_label1[0,idx]:
                probe_pairLabel_list.append(1)
            else:
                probe_pairLabel_list.append(0)

        if gallery_label1[0,minIndex] == probe_label1[0,idx]:

            ture_samples += 1

        if idx%1000==0 and idx>0:
            print("testing process:" + str(idx)+'/'+str(Probe_dis_List_matric1.shape[0])+'  |  '\
                  +"current acc: " +str(ture_samples/count) )

    acc = (float(ture_samples) / Probe_dis_List_matric1.shape[0])
    print('accuracy of testing set is %f%%' % acc)  # RANK-1 accuracy

    # TODO ROC
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc  ### roc, auc
    import matplotlib.pyplot as plt
    fpr, tpr, threshold = roc_curve(probe_pairLabel_list, probe_dis_List,drop_intermediate=False)  ### fpr, tpr
    roc_auc = auc(fpr, tpr)  ### auc

    for idx, val in enumerate(fpr):
        if val > 0.000001 and val < 0.0000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.00001 and val < 0.000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.0001 and val < 0.00019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.001 and val < 0.0019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.01 and val < 0.019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.1 and val < 0.19:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break



    ########################################################################
    # CASIA_WebFace: rank-1 and ROC results
    print('########################################################################')
    print('########################################################################')
    print('the backbone network is pre-trained in CASIA_WebFace and fine-tuned in Oulu-CASIA NIR-VIS')
    print('rank-1 and ROC results')
    print(len(Probe_dis_List_matric1[0]))

    ture_samples=0
    probe_dis_List=[]
    probe_pairLabel_list=[]


    count=0
    for idx in range(Probe_dis_List_matric1.shape[0]):


        net1_dis_vec=Probe_dis_List_matric1[idx,:]*net1
        net2_dis_vec = Probe_dis_List_matric2[idx, :] * net2
        net3_dis_vec = Probe_dis_List_matric3[idx, :] * net3
        net_dis_vec=net1_dis_vec+net2_dis_vec+net3_dis_vec

        minVals = 100000
        minIndex=-1
        count += 1
        # for index,val in enumerate(net_vec):
        ###according to dist
        for index, val in enumerate(net3_dis_vec):

            # probe_dis_List.append(-val)
            probe_dis_List.append(-net3_dis_vec[index])


            diff=val
            if diff < minVals:
                minIndex = index
                minVals=diff

            if gallery_label1[0,index] == probe_label1[0,idx]:
                probe_pairLabel_list.append(1)
            else:
                probe_pairLabel_list.append(0)

        if gallery_label1[0,minIndex] == probe_label1[0,idx]:

            ture_samples += 1

        if idx%1000==0 and idx>0:
            print("testing process:" + str(idx)+'/'+str(Probe_dis_List_matric1.shape[0])+'  |  '\
                  +"current acc: " +str(ture_samples/count) )

    acc = (float(ture_samples) / Probe_dis_List_matric1.shape[0])
    print('accuracy of testing set is %f%%' % acc)  # RANK-1 accuracy

    # TODO ROC
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc  ###roc, auc
    import matplotlib.pyplot as plt
    fpr, tpr, threshold = roc_curve(probe_pairLabel_list, probe_dis_List,drop_intermediate=False)  ### fpr, tpr
    roc_auc = auc(fpr, tpr)  ### auc

    for idx, val in enumerate(fpr):
        if val > 0.000001 and val < 0.0000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.00001 and val < 0.000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.0001 and val < 0.00019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.001 and val < 0.0019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.01 and val < 0.019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.1 and val < 0.19:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break

    ########################################################################
    # DC results rank-1 and ROC results
    print('########################################################################')
    print('########################################################################')
    print('The MDN network is pre-trained using three datasets and fine-tuned in Oulu-CASIA NIR-VIS. Finally, we adopt DC to fuse the three networks')
    print('rank-1 and ROC results')
    print(len(Probe_dis_List_matric1[0]))
    ture_samples=0
    probe_dis_List=[]
    probe_pairLabel_list=[]


    count=0
    for idx in range(Probe_dis_List_matric1.shape[0]):


        net1_dis_vec=Probe_dis_List_matric1[idx,:]*net1
        net2_dis_vec = Probe_dis_List_matric2[idx, :] * net2
        net3_dis_vec = Probe_dis_List_matric3[idx, :] * net3
        net_dis_vec=net1_dis_vec+net2_dis_vec+net3_dis_vec

        minVals = 100000
        minIndex=-1
        count += 1
        # for index,val in enumerate(net_vec):
        ###according to dist
        for index, val in enumerate(net_dis_vec):

            # probe_dis_List.append(-val)
            probe_dis_List.append(-net_dis_vec[index])


            diff=val
            if diff < minVals:
                minIndex = index
                minVals=diff

            if gallery_label1[0,index] == probe_label1[0,idx]:
                probe_pairLabel_list.append(1)
            else:
                probe_pairLabel_list.append(0)

        if gallery_label1[0,minIndex] == probe_label1[0,idx]:

            ture_samples += 1

        if idx%1000==0 and idx>0:
            print("testing process:" + str(idx)+'/'+str(Probe_dis_List_matric1.shape[0])+'  |  '\
                  +"current acc: " +str(ture_samples/count) )

    acc = (float(ture_samples) / Probe_dis_List_matric1.shape[0])
    print('accuracy of testing set is %f%%' % acc)  # Rank-1 accuracy

    # TODO ROC
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc  ### roc, auc
    import matplotlib.pyplot as plt
    fpr, tpr, threshold = roc_curve(probe_pairLabel_list, probe_dis_List,drop_intermediate=False)  ### fpr, tpr
    roc_auc = auc(fpr, tpr)  ### auc

    for idx, val in enumerate(fpr):
        if val > 0.000001 and val < 0.0000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.00001 and val < 0.000019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.0001 and val < 0.00019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.001 and val < 0.0019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.01 and val < 0.019:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break
    for idx, val in enumerate(fpr):
        if val > 0.1 and val < 0.19:
            print('fpr,tpr', fpr[idx], tpr[idx])
            break


if __name__ == '__main__':
    print('test result in Oulu-CASIA NIR-VIS dataset')
    main()
