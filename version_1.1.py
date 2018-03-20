import argparse
import sys
from  numpy import array
import skimage.io as io
import cv2
import tensorflow as tf
import xlrd
import time
import numpy as np

# useful_data_cut--2000
# 条缺 1758
# 圆缺 8956
# 未熔合 450

# 夹渣 70
# 裂纹 916
# 气孔 96
# 条孔 204
# 条渣 899
# 未焊透 284

IMAGE_SIZE = 448

NUM_IMAGES = 1915
VALIDATION_SIZE = 400
TEST_SIZE = 40
TEST_IMAGES_BEGIN = 1800
TEST_IMAGES_END = 1915

# NUM_IMAGES = 10000
# VALIDATION_SIZE = 2000
# TEST_SIZE = 200
# TEST_IMAGES_BEGIN = 12000
# TEST_IMAGES_END = 12500

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_EPOCHS = 10
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
DEFECT = "条缺"
SEED = 66478
NUM_LABELS = 2
LEARNING_RATE = 0.1
EVAL_FREQUENCY = 100
FLAGS = None
REMARKS = 18 #备注在表中列数
DEFECTS = 3 #缺陷在表中列数


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def load_photo(coll,num):
    crop_img = coll[num]
    image = cv2.resize(crop_img,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC)
    image = image.reshape(IMAGE_SIZE*IMAGE_SIZE)
    image = (image-PIXEL_DEPTH/2)/PIXEL_DEPTH
    image=image.reshape(IMAGE_SIZE,IMAGE_SIZE)
    print(num)
    return image


def load_label(xml_path):
    labels_xml = xlrd.open_workbook(xml_path)
    labels_table = labels_xml.sheets()[0]
    defect = labels_table.col_values(DEFECTS)[1:TEST_IMAGES_END+1]
    remark = labels_table.col_values(REMARKS)[1:TEST_IMAGES_END+1]
    labels = []
    for i in range(0,TEST_IMAGES_END):
        if defect[i]==DEFECT:
            labels.append(1)
        else:
            if str(remark[i]).find(DEFECT)!=-1:
                labels.append(1)
            else:
                labels.append(0)
    labels = array(labels)
    return labels

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def main(_):
    file_path = "/Volumes/TOSHIBA EXT/final/useful_data_cut"
    str = file_path + '/*.png'
    xml_path = "/Volumes/TOSHIBA EXT/final/useful_data_cut/useful_data_cut.xls"
    # file_path = "D:\\final\\useful_data"
    # str = file_path + '/*.png'
    # xml_path = "D:\\final\\useful_data\\useful_data.xlsx"
    labels = load_label(xml_path)
    coll = io.ImageCollection(str)
    images = []
    for i in range(TEST_IMAGES_END):
        images.append(load_photo(coll, i))
    images = array(images)
    images = images.reshape(TEST_IMAGES_END,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)

    validation_data = images[:VALIDATION_SIZE, ...]
    validation_labels = labels[:VALIDATION_SIZE]
    train_data = images[VALIDATION_SIZE:TEST_IMAGES_BEGIN, ...]
    train_labels = labels[VALIDATION_SIZE:TEST_IMAGES_BEGIN]
    test_data = images[TEST_IMAGES_BEGIN:TEST_IMAGES_END]
    test_labels = labels[TEST_IMAGES_BEGIN:TEST_IMAGES_END]
    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    train_data_node = tf.placeholder(
        data_type(),
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        data_type(),
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    #  {tf.global_variables_initializer().run()}
    conv1_weights = tf.Variable(tf.truncated_normal([3,3,NUM_CHANNELS,32],  # 3x3 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([32],dtype = data_type()))
    # conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32],  # 3x3 filter, depth 32.
    #                             stddev=0.1,
    #                             seed=SEED, dtype=data_type()))
    # conv2_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 64],  # 3x3 filter, depth 64.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv3_biases = tf.Variable(tf.zeros([64], dtype=data_type()))
    # conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64],  # 3x3 filter, depth 64.
    #                             stddev=0.1,
    #                             seed=SEED, dtype=data_type()))
    # conv4_biases = tf.Variable(tf.zeros([64], dtype=data_type()))
    conv5_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128],  # 3x3 filter, depth 128.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv5_biases = tf.Variable(tf.zeros([128], dtype=data_type()))
    # conv6_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128],  # 3x3 filter, depth 128.
    #                             stddev=0.1,
    #                             seed=SEED, dtype=data_type()))
    # conv6_biases = tf.Variable(tf.zeros([128], dtype=data_type()))
    conv7_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256],  # 3x3 filter, depth 256.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv7_biases = tf.Variable(tf.zeros([256], dtype=data_type()))
    # conv8_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],  # 3x3 filter, depth 256.
    #                             stddev=0.1,
    #                             seed=SEED, dtype=data_type()))
    # conv8_biases = tf.Variable(tf.zeros([256], dtype=data_type()))
    # conv9_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],  # 3x3 filter, depth 256.
    #                             stddev=0.1,
    #                             seed=SEED, dtype=data_type()))
    # conv9_biases = tf.Variable(tf.zeros([256], dtype=data_type()))
    conv10_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512],  # 3x3 filter, depth 512.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv10_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    # conv11_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
    #                              stddev=0.1,
    #                              seed=SEED, dtype=data_type()))
    # conv11_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    # conv12_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
    #                              stddev=0.1,
    #                              seed=SEED, dtype=data_type()))
    # conv12_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    conv13_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                 stddev=0.1,
                                 seed=SEED, dtype=data_type()))
    conv13_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    # conv14_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
    #                              stddev=0.1,
    #                              seed=SEED, dtype=data_type()))
    # conv14_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    # conv15_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
    #                              stddev=0.1,
    #                              seed=SEED, dtype=data_type()))
    # conv15_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 64 * IMAGE_SIZE // 64 * 512, 4096],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[4096], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[4096], dtype=data_type()))
    fc3_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([4096, 1000],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[1000], dtype=data_type()))
    fc4_weights = tf.Variable(tf.truncated_normal([1000, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc4_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=data_type()))


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv2_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # # Bias and rectified linear non-linearity.
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv3_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv4_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv5_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv6_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv6_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv7_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv7_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv8_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv8_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv9_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv9_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv10_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv10_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv11_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv11_biases))
        # conv = tf.nn.conv2d(relu,
        #                 conv12_weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding='SAME')
        # relu = tf.nn.relu(tf.nn.bias_add(conv, conv12_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv13_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv13_biases))

        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        pool_shape = pool.get_shape().as_list()
        pool_reshape = tf.reshape(pool,[pool_shape[0],pool_shape[1]*pool_shape[2]*pool_shape[3]])
        fc = tf.nn.relu(tf.matmul(pool_reshape,fc1_weights)+fc1_biases)
        fc = tf.nn.relu(tf.matmul(fc,fc2_weights)+fc2_biases)
        fc = tf.nn.relu(tf.matmul(fc, fc3_weights) + fc3_biases)
        if train:
            fc = tf.nn.dropout(fc,0.5,seed=SEED)
        return tf.nn.relu(tf.matmul(fc,fc4_weights)+fc4_biases)

    logits = model(train_data_node,True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=train_labels_node))
    regularizers = (tf.nn.l2_loss(fc1_weights)+tf.nn.l2_loss(fc1_biases)
                +tf.nn.l2_loss(fc2_weights)+tf.nn.l2_loss(fc2_biases)
                +tf.nn.l2_loss(fc3_weights)+tf.nn.l2_loss(fc3_biases)
                +tf.nn.l2_loss(fc4_weights)+tf.nn.l2_loss(fc4_biases))
    loss += 5e-4 * regularizers

    batch = tf.Variable(0,dtype=data_type())
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch*BATCH_SIZE,train_size,0.95,staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    def eval_in_batches(data,sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("Batch size for evals larger than dataset:%d"%size)
        predictions = np.ndarray(shape = (size,NUM_LABELS),dtype=np.float32)
        for begin in range(0,size,EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end < size:
                predictions = sess.run(eval_prediction,feed_dict={eval_data:data[begin:end,...]})
            else:
                batch_predictions = sess.run(eval_prediction,feed_dict={eval_data:data[-EVAL_BATCH_SIZE:,...]})
                predictions[begin-size:,:] = batch_predictions[begin-size:,:]
        return predictions


    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print("Initialized!!!")
        # Loop through training steps.
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE)%(train_size-BATCH_SIZE)
            batch_data = train_data[offset:(offset+BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset+BATCH_SIZE)]

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)

            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                l,lr,predictions = sess.run([loss,learning_rate,train_prediction],feed_dict=feed_dict)
                dur_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %(step, float(step) * BATCH_SIZE / train_size,
                       1000 * dur_time/ EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
                # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)
        # if FLAGS.self_test:
        #     print('test_error', test_error)
        #     assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
        #         test_error,)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)