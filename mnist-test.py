import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        data = f.read()
#        print(data)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(data)

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph



import argparse 
import tensorflow as tf

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="./mnist_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)


    x = graph.get_tensor_by_name('prefix/x:0')
    y = graph.get_tensor_by_name('prefix/output:0')

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    images, labels = mnist_test[0], mnist_test[1]
    images = images.astype('float32')
    images /= 255.0
    images = np.reshape(images, [10000, 28, 28, 1])
    print(images[0])

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        y_out = sess.run(y, feed_dict={
            x: [images[0]] # < 45
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(y_out) 
