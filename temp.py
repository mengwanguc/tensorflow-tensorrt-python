import tensorflow as tf
sess = tf.Session()
graph = tf.get_default_graph()
with tf.device('/gpu:0'):
  with graph.as_default():
    with sess.as_default():
        #restoring the model
        saver = tf.train.import_meta_graph('tmp/cifar10_model/model.ckpt-19540.meta')
        saver.restore(sess,tf.train.latest_checkpoint('tmp/cifar10_model'))
        #initializing all variables
        sess.run(tf.global_variables_initializer())
