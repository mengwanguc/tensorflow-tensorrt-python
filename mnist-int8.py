from tensorflow.python.client import session as csess
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.core.protobuf import config_pb2 as cpb2

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
precision_mode = 'INT8' # use 'FP32' for K80
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



# Use real data that is representative of the inference dataset
# for calibration. For this test script it is random data.
def run_calibration(gdef, dumm_inp):
  """Run given calibration graph multiple times."""
  gpu_options = None
  if trt.trt_convert.get_linked_tensorrt_version()[0] == 3:
    gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
  ops.reset_default_graph()
  gx = ops.Graph()
  with gx.as_default():
    inp, out = importer.import_graph_def(
        graph_def=gdef, return_elements=['x:0', 'output:0'])
    #inp = inp.outputs[0]
    #out = out.outputs[0]
  with csess.Session(
      config=cpb2.ConfigProto(gpu_options=gpu_options), graph=gx) as sess:
    # run over real calibration data here, we are mimicking a calibration set of
    # 30 different batches. Use as much calibration data as you want
    for j in range(200):
      val = sess.run(out, {inp: [dumm_inp[j]]})
  return val



mnist__train, temppp = tf.keras.datasets.mnist.load_data()
imagesss, labels = mnist__train[0], mnist__train[1]
imagesss = imagesss.astype('float32')
imagesss /= 255.0
imagesss = np.reshape(imagesss, [60000, 28, 28, 1])

_ = run_calibration(trt_graph_def, imagesss)
trt_graph_def=trt.calib_graph_to_infer_graph(trt_graph_def) # For only 'INT8'

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
