from styx_msgs.msg import TrafficLight
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import scipy.misc


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

    # normalize data
    x = tf.subtract(tf.div(x, 255.), 0.5)

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
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
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


class TLClassifier(object):
    def __init__(self, is_site):
        # TODO load classifier

        if is_site:
            num_classes = 4
        else:
            num_classes = 3

        image_shape = (192, 256)
        self.image_shape = image_shape
        sess = tf.Session()

        # with tf.Session() as sess:
        x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 3))
        keep_prob = tf.placeholder(tf.float32)  # probability to keep units
        logits = layers(x, keep_prob, num_classes)

        self.x = x
        self.logits = logits
        self.keep_prob = keep_prob

        saver = tf.train.Saver()

        if is_site:
            saver.restore(sess, tf.train.latest_checkpoint('./lenet_model_site/'))
        else:
            saver.restore(sess, tf.train.latest_checkpoint('./lenet_model/'))

        self.sess = sess

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        image = scipy.misc.imresize(image, self.image_shape)
        # with tf.Session() as sess:
        im_softmax = self.sess.run(
            [tf.argmax(self.logits, axis=1)],
            {self.keep_prob: 1.0, self.x: [image]})
        s = im_softmax[0]
        if s == 0:
            return TrafficLight.RED
        elif s == 1:
            return TrafficLight.YELLOW
        elif s == 2:
            return TrafficLight.GREEN
        elif s == 3:  # No traffic lights detected, return green to move on
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
