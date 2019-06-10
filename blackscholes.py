import tensorflow as tf

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

def N(param) :
    c = -1.0 / tf.sqrt(2.0);
    return tf.multiply(0.5, tf.erfc((c * param)))
  

p = (-1.0*S*N(-1.0*((tf.log(S/K) + (r + 0.5*v*v)*T) / (v*tf.sqrt(T)))) + K*tf.exp(-1.0*r*T)*N(-1.0*((tf.log(S/K) + (r + 0.5*v*v)*T) / (v*tf.sqrt(T)) - v*tf.sqrt(T))))
with tf.Session() as sess:
    print(sess.run(p))