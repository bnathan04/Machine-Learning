import numpy as np 
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import copy

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

trainData = np.reshape(trainData, (trainData.shape[0], -1))
validData = np.reshape(validData, (validData.shape[0], -1))
testData = np.reshape(testData, (testData.shape[0], -1))
trainTarget = np.reshape(trainTarget, (-1, 1))
validTarget = np.reshape(validTarget, (-1, 1))
testTarget = np.reshape(testTarget, (-1, 1))
num_categories = 10

# build a layer in NN
def build_layer(X, num_hidden_units):
    
    # initialize the weight matrix and bias vector
    num_inputs = X.get_shape().as_list()[-1]
    print("num hidden units:", num_hidden_units)
    W = tf.get_variable(name="Weights", shape=(num_inputs, num_hidden_units), dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(shape=(1, num_hidden_units), dtype=tf.float64, name="Bias"))

    # input to next layer
    z = tf.add(tf.matmul(X,W), b)

    return z

# Cross Entropy Loss calculation function
def calculate_ce_loss (truth, prediction, coeff):

    regularizer1 = (coeff / 2) * tf.reduce_sum(tf.square(tf.get_default_graph().get_tensor_by_name("hidden_layer1/Weights:0")))
    regularizer2 = (coeff / 2) * tf.reduce_sum(tf.square(tf.get_default_graph().get_tensor_by_name("hidden_layer2/Weights:0")))
    regularizer3 = (coeff / 2) * tf.reduce_sum(tf.square(tf.get_default_graph().get_tensor_by_name("softmax_layer/Weights:0")))
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=prediction))
    total_loss = ce_loss + regularizer1 + regularizer2 + regularizer3
    return total_loss

# set up hyper parameters 
num_train_steps = 15000
mini_batch_size = 500
weight_decay = 3e-4
num_data = trainData.shape[0]
num_epoch = int(math.ceil((num_train_steps * mini_batch_size)/num_data))
num_batches = num_data // mini_batch_size
num_hidden_units = 500
learning_rate = 0.001

# Set up place holders for the tf graph
X = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Data")
Y = tf.placeholder(tf.float64, shape=[None, 1], name="Label")

# Build the network using ReLU activation; 3 layers => two W matrices
with tf.variable_scope("hidden_layer1"):
    hidden_layer1 = tf.nn.relu(build_layer(X, num_hidden_units))

with tf.variable_scope("hidden_layer2"):
    hidden_layer2 = tf.nn.relu(build_layer(hidden_layer1, num_hidden_units))

with tf.variable_scope("softmax_layer"):    
    softmax_layer = tf.nn.relu(build_layer(hidden_layer2, num_categories))

# Classification
y_hat = tf.nn.softmax(softmax_layer)
is_correct = tf.equal(tf.expand_dims(tf.argmax(y_hat, 1), 1), tf.cast(Y, tf.int64))
error = 1 - (tf.reduce_mean(tf.cast(is_correct, tf.float32)))

# Loss calculations
ce_loss = calculate_ce_loss(tf.one_hot(tf.cast(Y, tf.int32), depth=10, axis=-1),
                                  softmax_layer, weight_decay)

# Empty arrays to record loss and error data
train_loss = np.zeros(num_epoch)
valid_loss = np.zeros(num_epoch)
test_loss = np.zeros(num_epoch)
train_err = np.zeros(num_epoch)
valid_err = np.zeros(num_epoch)
test_err = np.zeros(num_epoch)

# Graph x axis space
x_axis = [x+1 for x in range(num_epoch)]
fig_LR = plt.figure(1)
plt.title = "Error vs. Epoch"
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.grid(True)

# File setup
f = open("1_2_2_error.txt", "w+")

# Train
# Start session, (re)init variables and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ce_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for train_step in range(num_train_steps):
    
    # Get current batch data
    cur_batch_idx = (train_step % num_batches) * mini_batch_size
    cur_data = trainData[cur_batch_idx:cur_batch_idx + mini_batch_size]
    cur_target = trainTarget[cur_batch_idx:cur_batch_idx + mini_batch_size]
    optimizer_value = sess.run(optimizer, feed_dict={X: cur_data, Y: cur_target})

    # Every epoch, store the loss and error data

    cur_epoch = (((train_step + 1) * mini_batch_size) / num_data) - 1

    if ((train_step + 1) * mini_batch_size) % num_data == 0:
        
        # Get loss and error
        [train_loss[cur_epoch], train_err[cur_epoch]] = sess.run(fetches=[ce_loss, error], 
                                                            feed_dict={X: cur_data, Y: cur_target})

        [valid_loss[cur_epoch], valid_err[cur_epoch]] = sess.run(fetches=[ce_loss, error], 
                                                            feed_dict={X: validData, Y: validTarget})

        [test_loss[cur_epoch], test_err[cur_epoch]] = sess.run(fetches=[ce_loss, error], 
                                                            feed_dict={X: testData, Y: testTarget})

        print("---------- {} EPOCH(S) FINISHED AT {} = {} - Results ----------".format(cur_epoch + 1, 'H', num_hidden_units))
        print("Train loss:", train_loss[cur_epoch], "Valid loss:", valid_loss[cur_epoch], "Test loss:", test_loss[cur_epoch])
        print("Train error:",  train_err[cur_epoch], "Valid error:", valid_err[cur_epoch], "Test error:", test_err[cur_epoch])
        # print("Optimizer value: {}".format(optimizer_value))
        print("---------- END ----------")

        f.write("---------- %d EPOCH(S) FINISHED AT %s = %f - Results ----------\n" % (cur_epoch + 1, 'H', num_hidden_units))
        f.write("Train loss: %f, Valid loss: %f, Test loss: %f\n" % (train_loss[cur_epoch], valid_loss[cur_epoch],test_loss[cur_epoch]))
        f.write("Train error: %f, Valid error: %f, Test error: %f\n" % (train_err[cur_epoch], valid_err[cur_epoch],test_err[cur_epoch]))
        # f.write("Optimizer value: {}".format(optimizer_value))
        f.write("---------- END ----------\r\n")

# Choose best learning rate based using validation cross entropy loss as metric
print("END OF RUN for H: {}".format(num_hidden_units))
print("Results - final train loss: {}, final valid error: {}".format(train_loss[-1], valid_err[-1]))
plt.plot(x_axis, train_err, '-', label='Train Error')
plt.plot(x_axis, valid_err, '-', label='Validation Error')
plt.plot(x_axis, test_err, '-', label='Test Error')

plt.legend(loc="best")
fig_LR.savefig("1_2_2_error.png")
plt.show()
