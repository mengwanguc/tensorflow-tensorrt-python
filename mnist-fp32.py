import time
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
 
#
# tuning parameters
# (please change these values along with your computing resources)
#
batch_size = 128
workspace_size_bytes = 1 << 30
precision_mode = 'FP32' # use 'FP32' for K80
trt_gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction = 0.50)
 
#
# Read images (images -> input vectors)
#
tf.reset_default_graph()
g1 = tf.Graph()
""" Uncomment if you test batch inference """
# import numpy as np
# image1 = np.tile(image1,(batch_size,1,1,1))
# image2 = np.tile(image2,(batch_size,1,1,1))
# image3 = np.tile(image3,(batch_size,1,1,1))
print('Loaded image vectors (tiger, lion, orangutan)')
 
#
# Load classification graph def (pre-trained)
#
classifier_model_file = './mnist.pb' # downloaded from NVIDIA sample
classifier_graph_def = tf.GraphDef()
with tf.gfile.Open(classifier_model_file, 'rb') as f:
  data = f.read()
  classifier_graph_def.ParseFromString(data)
print('Loaded classifier graph def')
 
#
# Convert to TensorRT graph def
#
trt_graph_def = trt.create_inference_graph(
  input_graph_def=classifier_graph_def,
  outputs=['output'],
  max_batch_size=batch_size,
  max_workspace_size_bytes=workspace_size_bytes,
  precision_mode=precision_mode)
#trt_graph_def=trt.calib_graph_to_infer_graph(trt_graph_def) # For only 'INT8'
print('Generated TensorRT graph def')
 
#
# Generate tensor with TensorRT graph def
#
tf.reset_default_graph()
g2 = tf.Graph()
with g2.as_default():
  trt_x, trt_y = tf.import_graph_def(
    trt_graph_def,
    return_elements=['x:0', 'output:0'])
print('Generated tensor for TensorRT optimized graph')
 
#
# Run classification with TensorRT graph
#
with tf.Session(graph=g2, config=tf.ConfigProto(gpu_options=trt_gpu_ops)) as s2:
  #
  # predict image1 (tiger)
  #

  feed_dict = {
    trt_x: image1
  }
  start_time = time.time()
  result = s2.run([trt_y], feed_dict=feed_dict)
  stop_time = time.time()
  # list -> 1 x n ndarray : feature's format is [[1.16643378e-06 3.12126781e-06 3.39836406e-05 ... ]]
  nd_result = result[0]
  # remove row's dimension
  onedim_result = nd_result[0,]
  # set column index to array of possibilities 
  indexed_result = enumerate(onedim_result)
  # sort with possibilities
  sorted_result = sorted(indexed_result, key=lambda x: x[1], reverse=True)
  # get the names of top 5 possibilities
  for top in sorted_result[:5]:
    print(classes_entries[top[0]], 'confidence:', top[1])
  print('{:.2f} milliseconds'.format((stop_time-start_time)*1000))
  #
  # predict image2 (lion)
  #
  feed_dict = {
    trt_x: image2
  }
  start_time = time.time()
  result = s2.run([trt_y], feed_dict=feed_dict)
  stop_time = time.time()
  # list -> 1 x n ndarray : feature's format is [[1.16643378e-06 3.12126781e-06 3.39836406e-05 ... ]]
  nd_result = result[0]
  # remove row's dimension
  onedim_result = nd_result[0,]
  # set column index to array of possibilities 
  indexed_result = enumerate(onedim_result)
  # sort with possibilities
  sorted_result = sorted(indexed_result, key=lambda x: x[1], reverse=True)
  # get the names of top 5 possibilities
  for top in sorted_result[:5]:
    print(classes_entries[top[0]], 'confidence:', top[1])
  print('{:.2f} milliseconds'.format((stop_time-start_time)*1000))
  #
  # predict image3 (orangutan)
  #
  feed_dict = {
    trt_x: image3
  }
  start_time = time.time()
  result = s2.run([trt_y], feed_dict=feed_dict)
  stop_time = time.time()
  # list -> 1 x n ndarray : feature's format is [[1.16643378e-06 3.12126781e-06 3.39836406e-05 ... ]]
  nd_result = result[0]
  # remove row's dimension
  onedim_result = nd_result[0,]
  # set column index to array of possibilities 
  indexed_result = enumerate(onedim_result)
  # sort with possibilities
  sorted_result = sorted(indexed_result, key=lambda x: x[1], reverse=True)
  # get the names of top 5 possibilities
  for top in sorted_result[:5]:
    print(classes_entries[top[0]], 'confidence:', top[1])
  print('{:.2f} milliseconds'.format((stop_time-start_time)*1000))
