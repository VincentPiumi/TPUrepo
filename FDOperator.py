import tensorflow as tf
import os
from time import time

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

dx = 1
dy = 1
dz = 1

c1 = -1./24
c2 = 1./24
c3 = 9./8
c4 = -9./8

c1_ = [c1/dx, c1/dy, c1/dz]
c2_ = [c2/dx, c2/dy, c2/dz]
c3_ = [c3/dx, c3/dy, c3/dz]
c4_ = [c4/dx, c4/dy, c4/dz]

d1_ = [1, 1, 1]
d2_ = [2, 2, 2]
d3_ = [0, 0, 0]
d4_ = [1, 1, 1]

def _c1(d) : return tf.constant(c1_[d])
def _c2(d) : return tf.constant(c2_[d])
def _c3(d) : return tf.constant(c3_[d])
def _c4(d) : return tf.constant(c4_[d])

def _d1(d) : return tf.constant(d1_[d])
def _d2(d) : return tf.constant(d2_[d])
def _d3(d) : return tf.constant(d3_[d])
def _d4(d) : return tf.constant(d4_[d])

def run(tpu_computation, tpu_grpc_url) :

    reps = 1
    times = []

    for i in range(reps) :
        with tf.Session(tpu_grpc_url) as sess:
            sess.run(tpu.initialize_system())
            t1 = time()
            sess.run(tf.global_variables_initializer())
            sess.run(tpu_computation)
            t2 = time()
            print(str(i) + "_ : " + str(t2 - t1))
            times.append(t2 - t1)
            sess.run(tpu.shutdown_system())

    print(sum(times) / reps)

def apply_(fijk, i, j, DK) :
    c1 = _c1(0)
    c2 = _c2(0)
    c3 = _c3(0)
    c4 = _c4(0)

    d1 = _d1(0)
    d2 = _d2(0)
    d3 = _d3(0)
    d4 = _d4(0)

    size = fijk.get_shape()[2]

    slice1 = tf.slice(fijk, [i + d1, j, 0], [1, 1, size])
    slice2 = tf.slice(fijk, [i - d2, j, 0], [1, 1, size])
    slice3 = tf.slice(fijk, [i + d3, j, 0], [1, 1, size])
    slice4 = tf.slice(fijk, [i - d4, j, 0], [1, 1, size])

    fdo =  c1 * slice1 + c2 * slice2 + c3 * slice3 + c4 * slice4
    return fdo

if __name__ == "__main__" :

    dim1 = [0., 1., 2., 3., 4.]
    dim2 = [5., 6., 7., 8., 9.]

    dim3 = [10., 11., 12., 13., 14.]
    dim4 = [15., 16., 17., 18., 19.]

    fijk = tf.constant([[dim1, dim2, dim3, dim4], [dim2, dim3, dim4, dim1], [dim3, dim4, dim1, dim2], [dim4, dim1, dim2, dim3]])

    i = tf.constant(1)
    j = tf.constant(1)
    dk = tf.constant(0)

    inputs = [fijk, i, j, dk]

    tpu_computation = tpu.rewrite(apply_, inputs)
    tpu_grpc_url = TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()

    run(tpu_computation, tpu_grpc_url)
    print('Done !')
