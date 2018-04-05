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

num_train_steps = 15000
mini_batch_size = 500
num_data = trainData.shape[0]
num_epoch = int(math.ceil((num_train_steps * mini_batch_size)/num_data))
num_batches = num_data // mini_batch_size
num_hidden_units = [100, 500, 1000]

# build a layer in NN
def build_layer(input_layer, num_hidden_units):
    
    # initialize the weight matrix and bias vector
    num_inputs = input_layer.get_shape().as_list()[-1]
    W = tf.get_variable(name="Weights", shape=(num_inputs, num_hidden_units), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(shape=(1, num_hidden_units), dtype=tf.float32, name="Bias"))

    # input to next layer
    z = tf.add(tf.matmul(input_layer,W), b)

    return z

def buildGraph(num_hidden_units):

    # Set up
    num_categories = 10
    learning_rate = 0.001
    X = tf.placeholder(tf.float32, [None, 28, 28], name='data')
    X_flatten = tf.reshape(X, [-1, 28*28])
    Y = tf.placeholder(tf.float32, name='label')
    Y_onehot = tf.one_hot(tf.to_int64(Y), num_categories, 1.0, 0.0, axis = -1)
    weight_decay = 3e-4

    # Build network
    with tf.variable_scope("hidden_layer", reuse=True):
        hidden_layer = tf.nn.relu(build_layer(X_flatten, num_hidden_units))

    with tf.variable_scope("softmax_layer", reuse=True):    
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

    return X, Y, cross_entropy_loss, total_loss, class_error, train

# Empty arrays to record loss and error data
train_loss = np.zeros(num_epoch)
valid_loss = np.zeros(num_epoch)
test_loss = np.zeros(num_epoch)
train_err = np.zeros(num_epoch)
valid_err = np.zeros(num_epoch)
test_err = np.zeros(num_epoch)

best_train_err = []
best_train_loss = np.full((num_epoch), 99)
best_valid_err = []
best_valid_loss = []
best_test_err = []
best_test_loss = []
best_rate = 0

# Graph x axis space
x_axis = [x+1 for x in range(num_epoch)]
fig_LR = plt.figure(1)
plt.title = 'Validation Loss vs. Epoch'
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.grid(True)


# File setup
f = open("basic_NN_H_stats.txt", "w+")
# Train
for count, units in enumerate(num_hidden_units):

    # Start session, (re)init variables and optimizer
    # Training set up
    
    X, Y, ce_loss, total_loss, error, train = buildGraph(units)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for train_step in range(num_train_steps):
        
        # Get current batch data
        cur_batch_idx = (train_step % num_batches) * mini_batch_size
        cur_data = trainData[cur_batch_idx:cur_batch_idx + mini_batch_size]
        cur_target = trainTarget[cur_batch_idx:cur_batch_idx + mini_batch_size]
        optimizer_value = sess.run(train, feed_dict={X: cur_data, Y: cur_target})

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

            print("---------- {} EPOCH(S) FINISHED AT {} = {} - Results ----------".format(cur_epoch + 1, 'H', units))
            print("Train loss:", train_loss[cur_epoch], "Valid loss:", valid_loss[cur_epoch], "Test loss:", test_loss[cur_epoch])
            print("Train error:",  train_err[cur_epoch], "Valid error:", valid_err[cur_epoch], "Test error:", test_err[cur_epoch])
            print("---------- END ----------")

            f.write("---------- %d EPOCH(S) FINISHED AT %s = %f - Results ----------\n" % (cur_epoch + 1, 'H', units))
            f.write("Train loss: %f, Valid loss: %f, Test loss: %f\n" % (train_loss[cur_epoch], valid_loss[cur_epoch],test_loss[cur_epoch]))
            f.write("Train error: %f, Valid error: %f, Test error: %f\n" % (train_err[cur_epoch], valid_err[cur_epoch],test_err[cur_epoch]))
            f.write("---------- END ----------\r\n")

    # Choose best learning rate based using validation cross entropy loss as metric
    print("END OF RUN for H = {} units".format(units))
    print("Results - final valid loss: {}, final valid error: {}".format(valid_loss[-1], valid_err[-1]))
    plt.plot(x_axis, valid_loss, '-', str(units) + ' hidden units' )

    if count == 2: # after it was determined that 1000 was best num hidden units
        print("Get best training run: {}".format(count))
        best_rate = count
        best_valid_loss = copy.deepcopy(valid_loss)
        best_valid_err = copy.deepcopy(valid_err)
        best_train_loss = copy.deepcopy(train_loss)
        best_train_err = copy.deepcopy(train_err)
        best_test_loss = copy.deepcopy(test_loss)
        best_test_err = copy.deepcopy(test_err)

plt.legend(loc="best")
fig_LR.savefig("basic_NN_H_tune.png")
plt.show()

fig_error = plt.figure(2)
plt.title = 'Classfication Error vs. Epoch'
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.grid(True)

plt.plot(x_axis, best_train_err, '-', label=('Training'))
plt.plot(x_axis, best_valid_err, '-', label=('Validation'))
plt.plot(x_axis, best_test_err, '-', label=('Test'))

plt.legend(loc="best")
fig_error.savefig("basic_NN_H_error.png")
plt.show()

fig_loss = plt.figure(3)
plt.title = 'Cross Entropy Loss vs. Epoch'
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.plot(x_axis, best_train_loss, '-', label=('Training'))
plt.plot(x_axis, best_valid_loss, '-', label=('Validation'))
plt.plot(x_axis, best_test_loss, '-', label=('Test'))

plt.legend(loc="best")
fig_loss.savefig("basic_NN_H_loss.png")
plt.show()
