import tensorflow as tf
from tensorflow.python.client import device_lib

loc = device_lib.list_local_devices()
print([x.name for x in loc])
