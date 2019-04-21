
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf

import re
from tensorflow.python.platform import gfile
from six import iteritems


def SLloss(embeddings,class_num,sample_class,embedding_size,Deta=1.5):# 20180818 Deta (20190311: MDNDC paper SL loss)
    """Calculate the SL loss

    Args:
      embedding_size: the embeddings dimentions.
      sample_class: the number of samples each subject.
      class_num: the number of people each batch.
      embeddings: the extracting features from the network.
      Deta: the margin value

    Returns:
      the SL loss as a float tensor.
    """
    labelsIdx = []
    labelzero=[]
    for idx in range(class_num):
        for i in range(sample_class):
            labelsIdx.append(idx)
            labelzero.append(0)
    print('labelsIdx', labelsIdx)
    labelsIdx = tf.constant(labelsIdx)

    with tf.variable_scope('RQ_loss'):
        class_mean = tf.segment_mean(embeddings, labelsIdx, name='class_mean')
        all_class_center = tf.segment_mean(embeddings, labelzero, name='class_mean')
        val_Multiply_class=tf.constant(1/(class_num*class_num-class_num)) #(class_num*class_num+class_num)/2  ??
        val_Multiply_sample = tf.constant(1/(sample_class*class_num))
        print('all_class_center', all_class_center)
        print('class_mean',class_mean)
        sampleNum=0
        # Deta=0.5
        for classIdx in range(class_num):

            class_mean_single=tf.slice(class_mean,[classIdx,0],[1,embedding_size])

            for classIdx2 in range(classIdx+1,class_num):
                class_mean_single_other = tf.slice(class_mean, [classIdx2, 0], [1, embedding_size])
                class_mean_single_subtract = tf.subtract(class_mean_single, class_mean_single_other)
                class_mean_single_subtract_square = tf.square(class_mean_single_subtract)

                neg_dist = tf.reduce_sum(class_mean_single_subtract_square)
                Sb_loss = tf.add(tf.subtract(0.0, neg_dist), Deta)
                Sb_loss = tf.maximum(Sb_loss, 0.0)


                class_mean_single_subtract_square = tf.multiply(Sb_loss, val_Multiply_class)
                if classIdx==0 and classIdx2==1:
                    Sb = class_mean_single_subtract_square
                else:
                    Sb = tf.add(Sb,class_mean_single_subtract_square)
                print('classIdx2',classIdx,classIdx2)
            for sampleIdx in range(sample_class):


                sample_embeddings = tf.slice(embeddings, [sampleNum, 0], [1, embedding_size])
                class_embeddings_subtract = tf.subtract(sample_embeddings, class_mean_single)
                class_embeddings_subtract_square = tf.square(class_embeddings_subtract)
                pos_dist = tf.reduce_sum(class_embeddings_subtract_square)
                class_embeddings_subtract_square = tf.multiply(pos_dist, val_Multiply_sample)

                if sampleNum==0:
                    Sw = class_embeddings_subtract_square
                    print('Sw = Sw_Tmp',sampleNum)
                else:
                    Sw = tf.add(Sw, class_embeddings_subtract_square)
                    print('Sw = tf.add(Sw, Sw_Tmp)',sampleNum)

                sampleNum += 1

            print('class_mean_single', class_mean_single)
            print('class_mean_single_subtract', class_mean_single_subtract)


        print('Sw',Sw)
        print('Sb',Sb)
        # loss = tf.div(tf.trace(Sw), tf.trace(Sb))
        # loss = tf.divide(tf.reduce_sum(Sw), tf.reduce_sum(Sb))
        # pos_dist=tf.reduce_sum(Sw)
        # neg_dist=tf.reduce_sum(Sb)
        # loss = tf.subtract(pos_dist, neg_dist)
        # alpha=0.2
        # basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        # loss = tf.maximum(basic_loss, 0.0)

        loss = tf.add(Sw, Sb)
        print('loss', loss)

    return loss


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    tmp_loss = losses + [total_loss]
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))




def load_model_collection(model,fix_variables):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)


        # saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))

        saver = tf.train.Saver(fix_variables)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file