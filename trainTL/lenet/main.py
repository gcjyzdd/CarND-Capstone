#!/usr/bin/env python3
import os.path
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import helper
import warnings
from distutils.version import LooseVersion
import time
import datetime

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def layers(x, keep_prob, n_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 72x96x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 5), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(5))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 5, 7), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(7))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(21504, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(160, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def optimize(logits, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_operation)

    # validation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, train_op, loss_operation, accuracy_operation  # cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, acc_op, input_image,
             y, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):
        print("epoch {} ".format(epoch))
        loss = 0
        i = 0
        for image, label in get_batches_fn(batch_size):
            #
            print("{}, ".format(i), end='')
            i += 1
            sess.run(train_op,
                     feed_dict={input_image: image, y: label, keep_prob: 0.5, learning_rate: 1e-3})
            loss += sess.run(acc_op,
                             feed_dict={input_image: image, y: label, keep_prob: 0.5, learning_rate: 1e-3})
        print("\nAvg loss =  {} ".format(loss/i))


def run():
    num_classes = 3
    image_shape = (192, 256)
    data_dir = '../data_tl'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 3))
        y = tf.placeholder(tf.int32, (None))
        correct_label = tf.one_hot(y, num_classes)
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)  # probability to keep units

        logits = layers(x, keep_prob, num_classes)

        EPOCHS = 10
        BATCH_SIZE = 8

        logits, train_op, cross_entropy_loss, acc_op = optimize(logits, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, acc_op, x, y,
                 keep_prob, learning_rate)

        saver = tf.train.Saver()
        saver.save(sess, './mdl_' + st)
        print("Model saved.")
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, x)

        # OPTIONAL: Apply the trained model to a video


def runTest():
    num_classes = 3
    image_shape = (192, 256)
    data_dir = './data_tl'
    runs_dir = './runs'
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer7_out = load_vgg(sess, vgg_path)
        logits = layers(layer7_out, keep_prob, num_classes)

        y = tf.placeholder(tf.int32, (None))
        correct_label = tf.one_hot(y, num_classes)

        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(logits, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
    # runTest()
    # process_video()
