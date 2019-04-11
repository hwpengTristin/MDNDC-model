import tensorflow as tf
import NewPaper_validate_on_CASIA_NIR_VIS_2_0_eva as CASIA_eva

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#We first pre-train the network with different VIS datasets (MSCeleb1M, VGGFace2, CASIAWebFace), and then fine-tune the network with NIR-VIS dataset. We adopt  three networks to 
# extract NIR-VIS features and use them to train the DC. Finally, the joint Bayesian is used as classifier.
def main(args):
    dataset1='MSCeleb1M'
    dataset2='CASIAWebFace'
    dataset3='VGGFace2'

    print('concat')
    CASIA_eva.Nir_Vis_evaluate_Collect_3VIS_Datasets_operation_performance(VIS_Dataset1=dataset1,
                                                                        VIS_Dataset2=dataset2,
                                                                        VIS_Dataset3=dataset3,
                                                                        operation='concat')
    print('DC')
    CASIA_eva.Nir_Vis_evaluate_Collect_3VIS_Datasets_operation_performance(VIS_Dataset1=dataset1,
                                                                        VIS_Dataset2=dataset2,
                                                                        VIS_Dataset3=dataset3,
                                                                        operation='DC')
   
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
