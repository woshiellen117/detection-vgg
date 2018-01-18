import argparse
import sys
from  numpy import array
import skimage.io as io
import cv2
import tensorflow as tf
import xlrd
import time

IMAGE_SIZE = 448
NUM_IMAGES = 100
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 50
NUM_EPOCHS = 2
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
DEFECT = ""
SEED = 66478
NUM_LABELS = 2
LEARNING_RATE = 0.1
EVAL_FREQUENCY = 10

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def load_photo(coll,num):
    h,w = coll[num].shape[:2]
    a,c = 0,0
    for i in range(100,w-1):
        if coll[num][300,i+1]>250:
            a=i
            break
    if a == 0:
        a = w
    for i in range(100,h - 1):
        if coll[num][i, 300] > 250:
            c = i
            break
    if c == 0:
        c = h
    crop_img = coll[num][0: c, 0: a]
    image = cv2.resize(crop_img,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC)
    image = image.reshape(IMAGE_SIZE*IMAGE_SIZE)
    image = (image-PIXEL_DEPTH/2)/PIXEL_DEPTH
    image=image.reshape(IMAGE_SIZE,IMAGE_SIZE)
    print(num)
    # cv2.namedWindow('image', 0)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # print(a)
    # print(c)
    return image


def load_label(xml_path):
    labels_xml = xlrd.open_workbook(xml_path)
    labels_table = labels_xml.sheets()[0]
    defect = labels_table.col_values(3)[0:NUM_IMAGES]
    labels = []
    for i in range(NUM_IMAGES):
        if defect[i]==DEFECT:
            labels.append(1)
        elif defect[i] == '':
            labels.append(0)
            #无效图像
        else:
            labels.append(0)
    labels = array(labels)
    return labels


def main(_):
    file_path = "D:\\final\\thumb"
    str = file_path + '/*.png'
    xml_path = "D:\\final\\final.xlsx"
    coll = io.ImageCollection(str)
    images = []
    for i in range(NUM_IMAGES):
        images.append(load_photo(coll, i))
    train_images = array(images)
    train_images = train_images.reshape(NUM_IMAGES,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
    train_labels = load_label(xml_path)
    validation_data = train_images[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_images[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
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
    conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32],  # 3x3 filter, depth 32.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 64],  # 3x3 filter, depth 64.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv3_biases = tf.Variable(tf.zeros([64], dtype=data_type()))
    conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64],  # 3x3 filter, depth 64.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv4_biases = tf.Variable(tf.zeros([64], dtype=data_type()))
    conv5_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128],  # 3x3 filter, depth 128.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv5_biases = tf.Variable(tf.zeros([128], dtype=data_type()))
    conv6_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128],  # 3x3 filter, depth 128.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv6_biases = tf.Variable(tf.zeros([128], dtype=data_type()))
    conv7_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256],  # 3x3 filter, depth 256.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv7_biases = tf.Variable(tf.zeros([256], dtype=data_type()))
    conv8_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],  # 3x3 filter, depth 256.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv8_biases = tf.Variable(tf.zeros([256], dtype=data_type()))
    conv9_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],  # 3x3 filter, depth 256.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv9_biases = tf.Variable(tf.zeros([256], dtype=data_type()))
    conv10_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512],  # 3x3 filter, depth 512.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv10_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    conv11_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                 stddev=0.1,
                                 seed=SEED, dtype=data_type()))
    conv11_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    conv12_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                 stddev=0.1,
                                 seed=SEED, dtype=data_type()))
    conv12_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    conv13_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                 stddev=0.1,
                                 seed=SEED, dtype=data_type()))
    conv13_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    conv14_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                 stddev=0.1,
                                 seed=SEED, dtype=data_type()))
    conv14_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
    conv15_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                 stddev=0.1,
                                 seed=SEED, dtype=data_type()))
    conv15_biases = tf.Variable(tf.zeros([512], dtype=data_type()))
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
        conv = tf.nn.conv2d(relu,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv3_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        conv = tf.nn.conv2d(relu,
                        conv4_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv5_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases))
        conv = tf.nn.conv2d(relu,
                        conv6_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv6_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv5_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases))
        conv = tf.nn.conv2d(relu,
                        conv6_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv6_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv7_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv7_biases))
        conv = tf.nn.conv2d(relu,
                        conv8_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv8_biases))
        conv = tf.nn.conv2d(relu,
                        conv9_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv9_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv10_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv10_biases))
        conv = tf.nn.conv2d(relu,
                        conv11_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv11_biases))
        conv = tf.nn.conv2d(relu,
                        conv12_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv12_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        conv = tf.nn.conv2d(pool,
                        conv13_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv13_biases))
        conv = tf.nn.conv2d(relu,
                        conv14_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv14_biases))
        conv = tf.nn.conv2d(relu,
                        conv15_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv15_biases))
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
        pool_shape = pool.get_shape().aslist()
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
                       1000 * elapsed_time / EVAL_FREQUENCY))

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