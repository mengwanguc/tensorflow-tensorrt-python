import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
 
#
# tuning parameters
# (please change these values along with your computing resources)
#
batch_size = 128
workspace_size_bytes = 1 << 30
precision_mode = 'FP16' # use 'FP32' for K80
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
  mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
  images, labels = mnist_test[0], mnist_test[1]
  images = images.astype('float32')
  images /= 255.0
  images = np.reshape(images, [10000, 28, 28, 1])

  correct = 0
  total_time = 0
  time_num = 0
  for i in range(10000):
    feed_dict = {
      trt_x: [images[i]]
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
    top = sorted_result[0]
    if top[0] == labels[i]:
      correct += 1
    if i > 100:
      time_num += 1
      total_time += (stop_time-start_time)*1000
  print(correct)
  print('{:.2f} milliseconds'.format(total_time / float(time_num)))
