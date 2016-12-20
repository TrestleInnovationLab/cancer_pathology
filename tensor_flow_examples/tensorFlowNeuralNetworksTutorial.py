# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 18:56:20 2016

@author: casey
"""

# Tutorial: https://www.tensorflow.org/tutorials/mnist/beginners/

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# To show i-th image in training set use the following command
#plt.imshow(mnist.train.images[i,:].reshape(28,28))

# Linear Model: y = W.x + b
# y - labels
# x - images
# W - Weight matrix
# b - bias
x = tf.placeholder(tf.float32,[None, 784]) # 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# See documentation as to why we use softmax function
## Partition functions!
y = tf.nn.softmax(tf.matmul(x,W) + b)

# y_ is a placeholder which will have the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# Use cross-entropy function as cost function, not sure what's the point of the
# reduce's
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1] ))

# Define train_step operation as one step in a gradient descent algorithm,
# minimizing the cross_entropy with a learning rate of 0.5 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize variables operation
init = tf.global_variables_initializer()

# Start session and initialize variables
sess = tf.Session()
sess.run(init)

# Run training for 1000 iterations using stochastic gradient descent - selecting
# 100 random entries to use on each iteration
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x : batch_xs, y_ : batch_ys})

# Create vector of booleans checking whether the predicted label, y, matches
# the true label y_.
# Note: argmax( ,1) is used to select the most likely prediction.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Calculate accuracy, but first need to convert the booleans to floats
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

print(sess.run(accuracy, feed_dict={x : mnist.test.images, y_ : mnist.test.labels}))
    
