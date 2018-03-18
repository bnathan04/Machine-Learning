import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy

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

print(validData.shape)
print(validTarget.shape)


# Cross Entropy Loss calculation function
def calculate_ce_loss (W, truth, prediction):

    ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=prediction))
    return ce_loss


# Generate random values for the W matrix and bias term from a normal distribution
W = tf.Variable(tf.truncated_normal(shape=[trainData.shape[1], 1], stddev=0.1, dtype=tf.float64))
b = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, dtype=tf.float64))

# Set up place holders for the tf graph
X_train = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Train_Data")
Y_train = tf.placeholder(tf.float64, shape=[None, 1], name="Train_Label")
X_valid = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Valid_Data")
Y_valid = tf.placeholder(tf.float64, shape=[None, 1], name="Valid_Label")
data = tf.placeholder(tf.float64, name="Data")
labels = tf.placeholder(tf.float64, name="Labels")

# Set up parameters, including learning rate to be tuned, and set empty arrays to record values at each step of training
learning_rate = [0.005, 0.001, 0.0001]
batch_size = 500
num_train_steps = 5000
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_data
loss_train = np.zeros(num_train_steps)
loss_valid = np.zeros(num_train_steps)
acc_train = np.zeros(num_train_steps)
acc_test = np.zeros(num_train_steps)
acc_valid = np.zeros(num_train_steps)
best_acc_train = []
best_acc_valid = []
best_acc_test = []
best_rate = 0
best_loss_valid = np.full((num_train_steps), 99)
logistic_train_loss = []
logistic_train_acc = []

# Loss and accuracy calculations ----

# # Cross Entropy Move to second run
ce_loss_train = calculate_ce_loss(W, Y_train, tf.matmul(X_train, W) + b)
ce_loss_valid = calculate_ce_loss(W, Y_valid, tf.matmul(X_valid, W) + b)

# Classification with logistic regression
class_prediction = tf.sigmoid(tf.matmul(data, W) + b)
classify_logistic = tf.cast(tf.greater(class_prediction, 0.5), tf.float64)
is_correct_logistic = tf.reduce_sum(tf.cast(tf.equal(classify_logistic, tf.cast(labels, tf.float64)), tf.float64))
accuracy_logistic = tf.cast(is_correct_logistic, tf.float64) / tf.cast(tf.shape(classify_logistic)[0], tf.float64)

# Normal equation
X_transpose = tf.transpose(X_train)
w_least_squares = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X_transpose, X_train)), X_transpose), Y_train)

# Classification with linear regression
prediction_ls = tf.matmul(X_train, w_least_squares) + b
classify_linear_ls = tf.cast(tf.greater(prediction_ls, 0.5), tf.float64)
is_correct_linear_ls = tf.reduce_sum(tf.cast(tf.equal(classify_linear_ls, tf.cast(Y_train, tf.float64)), tf.float64))
accuracy_linear_ls = tf.cast(is_correct_linear_ls, tf.float64) / tf.cast(tf.shape(classify_linear_ls)[0], tf.float64)

# Graphs set up
plot_lines = ["b-", "r-", "-g"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)

fig_valid_loss = plt.figure(1)
fig_valid_loss.patch.set_facecolor('white')
axes1 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

# Train logistic regression

# Start session, (re)init variables and optimizer
sess = tf.Session()
for count, rate in enumerate(learning_rate):

    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(ce_loss_train)
    sess.run(tf.global_variables_initializer())

    for train_step in range(num_train_steps):

        # Get current batch data
        cur_batch_idx = (train_step % num_batches) * batch_size
        cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
        cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]

        # Optimize
        [optimizer_val, loss_train[train_step], loss_valid[train_step]] = \
            sess.run(fetches=[optimizer, ce_loss_train, ce_loss_valid],
                     feed_dict={X_train: cur_data, Y_train: cur_target, X_valid: validData, Y_valid: validTarget})

        # Get current accuracy calculations
        acc_train[train_step] = sess.run(accuracy_logistic, feed_dict={data: trainData, labels: trainTarget})
        acc_valid[train_step] = sess.run(accuracy_logistic, feed_dict={data: validData, labels: validTarget})
        acc_test[train_step] = sess.run(accuracy_logistic, feed_dict={data: testData, labels: testTarget})

        # Every batch, print out a summary
        if train_step % batch_size == 0:
            # print("Train accuracy:", acc_train[train_step], "Valid accuracy:", acc_valid[train_step],
            #       "Test accuracy:", acc_test[train_step])
            print("Validation loss =", loss_valid[train_step])

    # print("DONE ROUND")
    plot_name = "Learning Rate:" + str(rate)
    loss_valid_plot = plt.plot(plot_steps, loss_valid, plot_lines[count], label=plot_name)

    # Choose best learning rate based using validation cross entropy loss as metric
    if count == 0:
        best_rate = count
        best_loss_valid = copy.deepcopy(loss_valid)
        best_acc_train = copy.deepcopy(acc_train)
        best_acc_valid = copy.deepcopy(acc_valid)
        best_acc_test = copy.deepcopy(acc_test)

    if best_loss_valid[num_train_steps - 1] > loss_valid[num_train_steps - 1]:
        best_rate = count
        best_loss_valid = copy.deepcopy(loss_valid)
        best_acc_train = copy.deepcopy(acc_train)
        best_acc_valid = copy.deepcopy(acc_valid)
        best_acc_test = copy.deepcopy(acc_test)

    # Save training loss with LR = 0.001, to compare with logistic regression MSE loss later
    if rate == 0.001:
        logistic_train_loss = copy.deepcopy(loss_train)
        logistic_train_acc = copy.deepcopy(acc_train)

plt.legend(loc="best")
fig_valid_loss.savefig("2_1_3_LR_tune.png")
plt.show()

print("Best Learning Rate = ", learning_rate[best_rate])
print("logistic test/valid/train:", best_acc_train[num_train_steps-1], best_acc_valid[num_train_steps-1],
      best_acc_test[num_train_steps - 1])

fig_valid_acc_logistic = plt.figure(1)
fig_valid_acc_logistic.patch.set_facecolor('white')
axes2 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")

acc_train_plot = plt.plot(plot_steps, best_acc_train, plot_lines[0], label="Classification Accuracy")
acc_valid_plot = plt.plot(plot_steps, best_acc_valid, plot_lines[1], label="Validation Accuracy")
acc_test_plot = plt.plot(plot_steps, best_acc_test, plot_lines[2], label="Test Accuracy")

plt.legend(loc="best")
fig_valid_acc_logistic.savefig("2_1_3_logistic_acc.png")
plt.show()

# re-init variables and run least squares optimization equation
sess.run(tf.global_variables_initializer())
sess.run(w_least_squares, feed_dict={X_train: trainData, Y_train: trainTarget})

# Plot "normal equation" info
# final_normal_mse = sess.run(mse, feed_dict={data: trainData, labels: trainTarget})
lin_acc_train = sess.run(accuracy_linear_ls, feed_dict={X_train: trainData, Y_train: trainTarget})
lin_acc_valid = sess.run(accuracy_linear_ls, feed_dict={X_train: validData, Y_train: validTarget})
lin_acc_test = sess.run(accuracy_linear_ls, feed_dict={X_train: testData, Y_train: testTarget})

print("normal equation test/valid/train:", lin_acc_train, lin_acc_valid, lin_acc_test)
#
# # Linear regression set up for learning instead of least squares normal equation
# prediction = tf.matmul(X_train, W) + b
# mse = tf.reduce_mean(tf.square(prediction - Y_train)) / 2
# classify_linear = tf.cast(tf.greater(prediction, 0.5), tf.float64)
# is_correct_linear = tf.reduce_sum(tf.cast(tf.equal(classify_linear_ls, tf.cast(Y_train, tf.float64)), tf.float64))
# accuracy_linear = tf.cast(is_correct_linear, tf.float64) / tf.cast(tf.shape(classify_linear_ls)[0], tf.float64)
#
# # Train linear regression
# print("we out here !")
# optimizer_lin = tf.train.AdamOptimizer(learning_rate=learning_rate[1]).minimize(mse)
# sess.run(tf.global_variables_initializer())
#
# for train_step in range(num_train_steps):
#
#     # Get current batch data
#     cur_batch_idx = (train_step % num_batches) * batch_size
#     cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
#     cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]
#
#     # Optimize
#     sess.run(optimizer_lin, feed_dict={X_train: cur_data, Y_train: cur_target})
#
#     # Update loss and accuracy curves
#     loss_train[train_step] = sess.run(mse, feed_dict={X_train: cur_data, Y_train: cur_target})
#     acc_train[train_step] = sess.run(accuracy_linear, feed_dict={X_train: cur_data, Y_train: cur_target})
#
#     print("we out here?")
#     # Every batch, print out a summary
#     if train_step % 500 == 0:
#         print("Train accuracy:", acc_train[train_step], "Train loss:", loss_train[train_step])
#
# # Graphs set up
# # plot_lines = ["b-", "r-"]
# plot_steps = np.linspace(0, num_epoch, num=num_train_steps)
#
# fig_linvlog_loss = plt.figure(2)
# fig_linvlog_loss.patch.set_facecolor('white')
# axes2 = plt.gca()
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
#
# loss_plot_lin = plt.plot(plot_steps, loss_train, plot_lines[0], label="MSE")
# loss_plot_log = plt.plot(plot_steps, logistic_train_loss, plot_lines[1], label="Cross Entropy")
#
# plt.legend(loc="best")
# fig_linvlog_loss.savefig("2_1_3_linvlog_loss.png")
# plt.show()
#
# fig_linvlog_acc = plt.figure(3)
# fig_linvlog_acc.patch.set_facecolor('white')
# axes3 = plt.gca()
# plt.xlabel("Epoch")
# plt.ylabel("Training Accuracy")
#
# acc_plot_lin = plt.plot(plot_steps, acc_train, plot_lines[0], label="Linear Regression")
# acc_plot_log = plt.plot(plot_steps, logistic_train_acc, plot_lines[1], label="Logistic Regression")
#
# plt.legend(loc="best")
# fig_linvlog_acc.savefig("2_1_3_linvlog_acc.png")
# plt.show()
#
# print("We got here safely")
