# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 18:25:14 2016

@author: casey
"""

import tensorflow as tf

# Example of matrix multiplication

## matrix1 is row vector (3,5)
matrix1 = tf.constant([[3., 5.]])
## matrix2 is a column vector, (2,4)^T
matrix2 = tf.constant([[2.], [4.]])

## product1 is the inner product producing a scalar
product1 = tf.matmul(matrix1, matrix2)
## product2 is the tensor/outer product, producing a 2x2 matrix
product2 = tf.matmul(matrix2, matrix1)

## Create and run a new session as sess
with tf.Session() as sess:
    ## Run inner product operation, store it in result1 and print value
    result1 = sess.run([product1])
    print(result1)
    ## Run tensor product operation, store it in result2 and print value
    result2 = sess.run([product2])
    print(result2)
    

# Working with variables and constants

## Define a variable state, and initialize to 0, not sure what the name thing
## is about.
state = tf.Variable(0, name="counter")
## Define a constant object "one" and initialize to 1
one = tf.constant(1)
## Define the variable new_value to be the result of the .add operation of
## variables state and one
new_value = tf.add(state,one)
## Define the update operation to be assigning state to equal new_value
update = tf.assign(state, new_value)
## Define the initialize all variables operation
init_op = tf.global_variables_initializer()
## Create and run new session
with tf.Session() as sess:
    ## Initialize all variables
    sess.run(init_op)
    ## This prints the result of the state operation, which is assignment to 0
    print(sess.run(state))
    ## Now loop through and update the state variable
    for _ in range(3):
        ## run update operation
        sess.run(update)
        ## print result
        print(sess.run(state))

# This example shows how to run multiple operations at the same time

## Define some constants
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
## define two operations add, and multiply
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)
## Run Session
with tf.Session() as sess:
    ## Note that we just put the mul and intermed operations as an array
    result = sess.run([mul, intermed])
    print(result)

# The following highlights the use of feeds, essentially objects that will be 
# supplied at runtime.

## inputs 4 & 5 will be dummy variables to be used to stand in for floats that
## will be supplied at runtime    
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
## output is defined to be the multiplication operation
output = tf.mul(input4, input5)
## Run session
with tf.Session() as sess:
    ## To supply the feeds, use the feed_dict syntax below
    print(sess.run([output], feed_dict = {input4 : [7.] , input5 : [9.]}))