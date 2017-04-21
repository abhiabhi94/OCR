import tensorflow as tf, sys

# change this as you see fit
image_path = '/home/abhyudai/sample/a.jpg'
def recognize():
    #image_path = "/home/abhyudai/Desktop/OCR/samples/test_C1.jpg"
    # print 0

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # print type(image_data)

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("/home/abhyudai/Desktop/tensorflow/tf_files/retrained_labels.txt")]

    # print 1

    # Unpersists graph from file
    with tf.gfile.FastGFile("/home/abhyudai/Desktop/tensorflow/tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        # print 2

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # print 3
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        # for node_id in top_k:
        #     human_string = label_lines[node_id]
        #     score = predictions[0][node_id]
        #     print('%s (score = %.5f)' % (human_string, score))
        # print 4

        return label_lines[top_k[0]], predictions[0][top_k[0]]

# print recognize("/home/abhyudai/sample/a.jpg")
print recognize()