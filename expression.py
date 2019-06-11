import os
import tensorflow as tf
from time import time

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

def expression(a, b, c):
	ad = tf.add(a, b)
	co = tf.multiply(2.4, c)
	su = tf.subtract(ad, co)
	lo = tf.log(su)
	sq = tf.sqrt(lo)
	return sq

def run(size):
	a_ = []
	b_ = []
	c_ = []
	for i in range(size) :
		a_.append((i * 1.0 + 4.0) * 2.5)
		b_.append((i * 1.0 + 5.0) * 2.5)
		c_.append((i * 1.0 + 6.0) * 0.1)

	inputs = [tf.constant(a_), tf.constant(b_), tf.constant(c_)]

	tpu_computation = tpu.rewrite(expression, inputs)
	tpu_grpc_url = TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()

	with tf.Session(tpu_grpc_url) as sess:
  		sess.run(tpu.initialize_system())
  		t1 = time()
		sess.run(tf.global_variables_initializer())
  		sess.run(tpu_computation)
		t2 = time()
		print(str(size) + " : " + str(t2 - t1))
		sess.run(tpu.shutdown_system())

	print('Done !')

if __name__ == "__main__" :
	run(1000)
	run(10000)
	run(100000)
	run(1000000)
	run(5000000)
	run(10000000)

