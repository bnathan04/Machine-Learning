import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


# load data
with np.load("notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target == posClass) + (Target == negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target == posClass] = 1
    Target[Target == negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

# Reshape the data to fit the linear regression model
trainData = np.reshape(trainData, (3500, -1))
validData = np.reshape(validData, (100, -1))
testData = np.reshape(testData, (145, -1))

# Generate random values for the W matrix and bias term from a normal distribution
W = tf.Variable(tf.truncated_normal(shape=[trainData.shape[1], 1], stddev=0.1, dtype=tf.float64))
b = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, dtype=tf.float64))

# Set up place holders for the tf graph
X = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Data")
Y = tf.placeholder(tf.float64, shape=[None, 1], name="Label")
# Lambda = tf.placeholder(tf.float64, shape=[1, ], name="Weight_Decay")

# Set loss function parameters
learning_rate = 0.005
# weight_decay_coeff = np.array([0.0, 0.001, 0.1, 1])
# weight_decay_coeff = np.reshape(weight_decay_coeff, (4, -1))

# Prepare for training runs
num_train_steps = 20000
batch_size = 500
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_data

loss_train = np.zeros(num_train_steps)
acc_train = np.zeros(num_train_steps)
# acc_valid = np.zeros(num_train_steps)

# Set up the loss calculations
prediction = tf.matmul(X, W) + b
# scale_factor = tf.constant(0.5, dtype=tf.float64)
# regularizer = tf.multiply(tf.multiply(tf.reduce_sum(tf.square(W)), Lambda), scale_factor)
mse = tf.reduce_mean(tf.square(prediction - Y)) / 2  # + regularizer
classify = tf.cast(tf.greater(prediction, 0.5), tf.float64)
is_correct = tf.reduce_sum(tf.cast(tf.equal(classify, tf.cast(Y, tf.float64)), tf.float64))
accuracy = tf.cast(is_correct, tf.float64) / tf.cast(tf.shape(classify)[0], tf.float64)

# Graphs set up
# plot_lines = ["b-", "r-"]
# plot_steps = np.linspace(0, num_epoch, num=num_train_steps)
#
# fig_loss = plt.figure()
# fig_loss.patch.set_facecolor('white')
# axes1 = plt.gca()
# ax1 = fig_loss.add_subplot(111)
# ax1.xlabel("Epoch")
# ax1.ylabel("Loss")
#
# fig_acc = plt.figure()
# fig_acc.patch.set_facecolor('white')
# axes2 = plt.gca()
# ax2 = fig_acc.add_subplot(111)
# ax2.xlabel("Epoch")
# ax2.ylabel("Accuracy")


# Train
# Start tf session, init variables and SGD optimizer function
sess = tf.Session()
sess.run(tf.global_variables_initializer())
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

# Run training for SGD
start_time = time.time()
for step in range(num_train_steps):
    cur_batch_idx = (step % num_batches) * batch_size
    cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
    cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]
    sess.run(optimizer, feed_dict={X: cur_data, Y: cur_target})

    loss_train[step] = sess.run(mse, feed_dict={X: cur_data, Y: cur_target})
    acc_train[step] = sess.run(accuracy, feed_dict={X: cur_data, Y: cur_target})
    # if step % batch_size == 0:
    #     print(rate, step, loss_train[step])

elapsed_time = time.time() - start_time
# Plot SGD info
final_sgd_mse = loss_train[num_train_steps-1]
sgd_plot_name = "SGD Optimization"
# sgd_loss_plot = ax1.plot(plot_steps, loss_train, plot_lines[0], label=sgd_plot_name)
final_sgd_acc = acc_train[num_train_steps - 1]
# sgd_acc_plot = ax2.plot(plot_steps, acc_train, plot_lines[0], label=sgd_plot_name)

print(sgd_plot_name, "- Final training accuracy =", final_sgd_acc, "Final MSE =", final_sgd_mse, "Elapsed Time =",
      elapsed_time)

# Normal equation
X_new = tf.concat([tf.add(tf.zeros([3500, 1], dtype=tf.float64), 1), X], 1)
print(sess.run(X_new, feed_dict={X: trainData}))
X_transpose = tf.transpose(X_new)
w_least_squares = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X_transpose, X_new)), X_transpose), Y)
prediction_ls = tf.matmul(X_new, w_least_squares) + b
mse_ls = tf.reduce_mean(tf.square(prediction_ls - Y)) / 2
classify_ls = tf.cast(tf.greater(prediction_ls, 0.5), tf.float64)
is_correct_ls = tf.reduce_sum(tf.cast(tf.equal(classify_ls, tf.cast(Y, tf.float64)), tf.float64))
accuracy_ls = tf.cast(is_correct_ls, tf.float64) / tf.cast(tf.shape(classify_ls)[0], tf.float64)


start_time = time.time()
# re-init variables and run least squares optimization equation
sess.run(tf.global_variables_initializer())
# sess.run(w_least_squares, feed_dict={X: trainData, Y: trainTarget})

# Plot "normal equation" info
final_normal_mse = sess.run(mse_ls, feed_dict={X: trainData, Y: trainTarget})
elapsed_time = time.time() - start_time
normal_plot_name = "Normal Equation Optimization"
# normal_loss_plot = ax1.plot(plot_steps, loss_train, plot_lines[0], label=plot_name)
final_normal_acc = sess.run(accuracy_ls, feed_dict={X: trainData, Y: trainTarget})
# normal_acc_plot = ax2.plot(plot_steps, acc_train, plot_lines[0], label=plot_name)

print(normal_plot_name, "- Final training accuracy =", final_normal_acc, "Final MSE =", final_normal_mse,
      "Elapsed Time =", elapsed_time)

# Save graphs
# plt.legend(loc="best")
# fig_loss.savefig("1_4_loss.png")
# fig_acc.savefig("1_4_acc.png")
# ax1.show()
# ax2.show()



# print("Done")







