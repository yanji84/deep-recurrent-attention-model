import tensorflow as tf
import numpy as np


class BaseNet(object):

    def __init__(self):
        return

    def variableOnCpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def variableWithWeightDecay(self, name, shape, stddev, wd):
        var = self.variableOnCpu(
            name, shape, tf.truncated_normal_initializer(
                stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def activationSummary(self, x):
        tf.histogram_summary(x.op.name + '/activations', x)
        tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

