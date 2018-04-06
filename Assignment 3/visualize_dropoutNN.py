import numpy as np 
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import copy
from PIL import Image

with np.load("notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

num_train_steps = 15000
mini_batch_size = 500
num_data = trainData.shape[0]
num_epoch = int(math.ceil((num_train_steps * mini_batch_size)/num_data))
num_batches = num_data // mini_batch_size

# build a layer in NN
def build_layer(input_layer, num_hidden_units):
    
    # initialize the weight matrix and bias vector
    num_inputs = input_layer.get_shape().as_list()[-1]
    W = tf.get_variable(name="Weights", shape=(num_inputs, num_hidden_units), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(shape=(1, num_hidden_units), dtype=tf.float32, name="Bias"))

    # input to next layer
    z = tf.add(tf.matmul(input_layer,W), b)

    return z

def buildGraph():

    # Set up
    num_hidden_units = 1000
    num_categories = 10
    learning_rate = 0.001
    X = tf.placeholder(tf.float32, [None, 28, 28], name='data')
    X_flatten = tf.reshape(X, [-1, 28*28])
    Y = tf.placeholder(tf.float32, name='label')
    Y_onehot = tf.one_hot(tf.to_int64(Y), num_categories, 1.0, 0.0, axis = -1)
    P = tf.placeholder(tf.float32)
    weight_decay = 3e-4

    # Build network
    with tf.variable_scope("hidden_layer"):
        hidden_layer = tf.nn.dropout(tf.nn.relu(build_layer(X_flatten, num_hidden_units)), P)

    with tf.variable_scope("softmax_layer"):    
        softmax_layer = tf.nn.relu(build_layer(hidden_layer, num_categories))

    # Calculate loss and error
    Y_predicted = tf.nn.softmax(softmax_layer)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                            labels=Y_onehot, logits=softmax_layer),
                                            name='cross_entropy_loss')
    class_error = 1 - (tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(Y_predicted, -1), tf.to_int64(Y)))))                                        
    
    W1 = tf.get_default_graph().get_tensor_by_name("hidden_layer/Weights:0")
    W2 = tf.get_default_graph().get_tensor_by_name("softmax_layer/Weights:0")
    regularizer = (tf.reduce_sum(W1*W1) * weight_decay * 0.5) + (tf.reduce_sum(W2*W2) * weight_decay * 0.5)
    total_loss = cross_entropy_loss + regularizer
    
    # Set up training
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss=total_loss)    

    return X, Y, P, cross_entropy_loss, class_error, train

def visualization(W, percent):

    f, axes = plt.subplots(10, 10, sharex='col', sharey='row')

    for j in range(10):
        for k in range(10):
            cur_W = W[:,:, 10*j+k]
            cur_W.reshape(cur_W.shape[0], cur_W.shape[1])
            axes[j, k].imshow(cur_W, plt.cm.gray)
            axes[j, k].get_xaxis().set_visible(False)
            axes[j, k].get_yaxis().set_visible(False)

    f.savefig("1_4_2_dropout" + str(percent) + ".png")
    plt.show()

# Empty arrays to record loss and error data
train_loss = np.zeros(num_epoch)
valid_loss = np.zeros(num_epoch)
test_loss = np.zeros(num_epoch)
train_err = np.zeros(num_epoch)
valid_err = np.zeros(num_epoch)
test_err = np.zeros(num_epoch)

# File setup
f = open("dropout_stats.txt", "w+")

# Train
X, Y, P, ce_loss, error, train = buildGraph()
saver = tf.train.Saver()
early_stop = 228
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for train_step in range(num_train_steps):
    
    # Get current batch data
    cur_batch_idx = (train_step % num_batches) * mini_batch_size
    cur_data = trainData[cur_batch_idx:cur_batch_idx + mini_batch_size]
    cur_target = trainTarget[cur_batch_idx:cur_batch_idx + mini_batch_size]
    optimizer_value = sess.run(train, feed_dict={X: cur_data, Y: cur_target, P: 0.5})

    # Every epoch, store the loss and error data
    cur_epoch = (((train_step + 1) * mini_batch_size) / num_data) - 1
    if ((train_step + 1) * mini_batch_size) % num_data == 0:
        # Get loss and error
        [train_loss[cur_epoch], train_err[cur_epoch]] = sess.run(fetches=[ce_loss, error], 
                                                            feed_dict={X: cur_data, Y: cur_target, P: 1.0})

        [valid_loss[cur_epoch], valid_err[cur_epoch]] = sess.run(fetches=[ce_loss, error], 
                                                            feed_dict={X: validData, Y: validTarget, P: 1.0})

        [test_loss[cur_epoch], test_err[cur_epoch]] = sess.run(fetches=[ce_loss, error], 
                                                            feed_dict={X: testData, Y: testTarget, P: 1.0})

        print("---------- {} EPOCH(S) FINISHED - Results ----------".format(cur_epoch + 1))
        print("Train loss:", train_loss[cur_epoch], "Valid loss:", valid_loss[cur_epoch], "Test loss:", test_loss[cur_epoch])
        print("Train error:",  train_err[cur_epoch], "Valid error:", valid_err[cur_epoch], "Test error:", test_err[cur_epoch])
        print("---------- END ----------")

        f.write("---------- %d EPOCH(S) FINISHED - Results ----------\n" % (cur_epoch + 1))
        f.write("Train loss: %f, Valid loss: %f, Test loss: %f\n" % (train_loss[cur_epoch], valid_loss[cur_epoch],test_loss[cur_epoch]))
        f.write("Train error: %f, Valid error: %f, Test error: %f\n" % (train_err[cur_epoch], valid_err[cur_epoch],test_err[cur_epoch]))
        f.write("---------- END ----------\r\n")
    
    if ((cur_epoch + 1) == int(0.25 *early_stop)):
        saver.save(sess, './dropout_NN_0.25',global_step=25)
    
    if (cur_epoch +1) == early_stop:
        break

# Choose best learning rate based using validation cross entropy loss as metric
print("END OF RUN")
print("Results - final valid loss: {}, final valid error: {}".format(valid_loss[-1], valid_err[-1]))

W = tf.reshape((tf.get_default_graph().get_tensor_by_name("hidden_layer/Weights:0")), [28, 28, 1000])
W_in = sess.run(W)
visualization(W_in, 100)

saver.restore(sess,tf.train.latest_checkpoint('./'))
W = tf.reshape((tf.get_default_graph().get_tensor_by_name("hidden_layer/Weights:0")), [28, 28, 1000])
W_in = sess.run(W)
visualization(W_in, 25)
