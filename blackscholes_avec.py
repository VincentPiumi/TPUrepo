import tensorflow as tf
import os
from time import time

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

def N(param) :
	c = -1.0 / tf.sqrt(2.0);
	return tf.multiply(0.5, tf.erfc((c * param)))

def blackscholes(S, K, r, T, v) :
	p = (-1.0*S*N(-1.0*((tf.log(S/K) + (r + 0.5*v*v)*T) / (v*tf.sqrt(T)))) + K*tf.exp(-1.0*r*T)*N(-1.0*((tf.log(S/K) + (r + 0.5*v*v)*T) / (v*tf.sqrt(T)) - v*tf.sqrt(T))))
	return p

def timer(inputs) :
	reps = 2
	times = []

	for i in range(reps) :
		t1 = time()
		tpu_computation = tpu.rewrite(blackscholes, inputs)
		tpu_grpc_url = TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()

		with tf.Session(tpu_grpc_url) as sess:
    			sess.run(tpu.initialize_system())
			sess.run(tf.global_variables_initializer())
			sess.run(tpu_computation)
			sess.run(tpu.shutdown_system())

		t2 = time()
		print(str(i) + "_ : " + str(t2 - t1))
		times.append(t2 - t1)

	print(sum(times) / reps)

def run() :
	S0=100.;
	K0=100.;
	r0=0.05;
	T0=1.0;
	v0=0.2;

	S = tf.constant(S0);
	K = tf.constant(K0);
	r = tf.constant(r0);
	T = tf.constant(T0);
	v = tf.constant(v0);

	inputs = [S, K, r, T, v]

	timer(inputs)
	print('Done !')

if __name__ == "__main__" :
	run()
