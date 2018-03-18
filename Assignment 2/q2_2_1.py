import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]


# Cross Entropy Loss calculation function
def calculate_ce_loss (W, truth, prediction, coeff):

    regularizer = (coeff / 2) * tf.reduce_sum(tf.square(W))
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=prediction))
    total_loss = ce_loss + regularizer
    return total_loss


# Reshape the data
trainData = np.reshape(trainData, (trainData.shape[0], 28*28))
trainTarget = np.reshape(trainTarget, (trainTarget.shape[0])).astype(int)
validData = np.reshape(validData, (validData.shape[0], 28*28))
validTarget = np.reshape(validTarget, (validTarget.shape[0])).astype(int)
testData = np.reshape(testData, (testData.shape[0], 28*28))
testTarget = np.reshape(testTarget, (testTarget.shape[0])).astype(int)

# Generate random values for the W matrix and bias term from a normal distribution
W = tf.Variable(tf.truncated_normal(shape=[trainData.shape[1], 10], stddev=0.1, dtype=tf.float64))
b = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, dtype=tf.float64))

# Set up place holders for the tf graph
X_train = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Train_Data")
Y_train = tf.placeholder(tf.float64, shape=[None], name="Train_Label")
X_valid = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Valid_Data")
Y_valid = tf.placeholder(tf.float64, shape=[None], name="Valid_Label")
data = tf.placeholder(tf.float64, name="Data")
labels = tf.placeholder(tf.int32, shape=[None], name="Labels")

# Set up parameters, including learning rate to be tuned, and set empty arrays to record values at each step of training
weight_decay = 0.01
learning_rate = [0.005, 0.001, 0.0001]
batch_size = 500
num_train_steps = 5000
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_data
num_categories = 10
loss_train = np.zeros(num_train_steps)
loss_valid = np.zeros(num_train_steps)
acc_train = np.zeros(num_train_steps)
acc_valid = np.zeros(num_train_steps)
acc_test = np.zeros(num_train_steps)
best_acc_train = []
best_loss_train = []
best_acc_valid = []
best_loss_valid = np.full((num_train_steps), 99)
best_acc_test = []
best_rate = 0

# Loss calculations
prediction_train = tf.matmul(X_train, W) + b
prediction_valid = tf.matmul(X_valid, W) + b
ce_loss_train = calculate_ce_loss(W, tf.one_hot(tf.cast(Y_train, tf.int32), depth=10, axis=-1),
                                  prediction_train, weight_decay)
ce_loss_valid = calculate_ce_loss(W, tf.one_hot(tf.cast(Y_valid, tf.int32), depth=10, axis=-1),
                                  prediction_valid, weight_decay)

# Classification
y_hat = tf.nn.softmax(tf.matmul(data, W) + b)
is_correct = tf.equal(tf.argmax(y_hat, 1), tf.cast(labels, tf.int64))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Graph set up
plot_lines = ["b-", "r-", "-g"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)

fig_valid_loss = plt.figure(1)
fig_valid_loss.patch.set_facecolor('white')
axes1 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

sess = tf.Session()

# Train --- iterate throuhg all possible learning rates
for count, rate in enumerate(learning_rate):

    # Start session, (re)init variables and optimizer
    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(ce_loss_train)
    sess.run(tf.global_variables_initializer())

    # Run training loop
    for train_step in range(num_train_steps):

        # Get current batch data
        cur_batch_idx = (train_step % num_batches) * batch_size
        cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
        cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]

        # Optimize and get the current loss calculations
        [optimizer_value, loss_train[train_step], loss_valid[train_step]] = \
            sess.run(fetches=[optimizer, ce_loss_train, ce_loss_valid],
                     feed_dict={X_train: cur_data, Y_train: cur_target, X_valid: validData, Y_valid: validTarget})

        # Get current accuracy calculations
        acc_train[train_step] = sess.run(accuracy, feed_dict={data: trainData, labels: trainTarget})
        acc_valid[train_step] = sess.run(accuracy, feed_dict={data: validData, labels: validTarget})
        acc_test[train_step] = sess.run(accuracy, feed_dict={data: testData, labels: testTarget})

        # Every batch, print out a summary
        # if train_step % batch_size == 0:
        #     print("Train loss:", loss_train[train_step], "Valid loss:", loss_valid[train_step],
        #           "Train accuracy:", acc_train[train_step], "Valid accuracy:", acc_valid[train_step],
        #           "Test accuracy:", acc_test[train_step])

    plot_name = "Learning Rate:" + str(rate)
    loss_valid_plot = plt.plot(plot_steps, loss_valid, plot_lines[count], label=plot_name)

    # Choose best learning rate based using validation cross entropy loss as metric
    if count == 0:
        best_rate = count
        best_loss_valid = copy.deepcopy(loss_valid)
        best_acc_valid = copy.deepcopy(acc_valid)
        best_loss_train = copy.deepcopy(loss_train)
        best_acc_train = copy.deepcopy(acc_train)
        best_acc_test = copy.deepcopy(acc_test)

    if best_loss_valid[num_train_steps-1] > loss_valid[num_train_steps - 1]:
        best_rate = count
        best_loss_valid = copy.deepcopy(loss_valid)
        best_acc_valid = copy.deepcopy(acc_valid)
        best_loss_train = copy.deepcopy(loss_train)
        best_acc_train = copy.deepcopy(acc_train)
        best_acc_test = copy.deepcopy(acc_test)

# Print out summary of training results
print("Best Learning Rate = ", learning_rate[best_rate])
print("Soft-Max Accuracy (training/validation/test):", best_acc_train[num_train_steps-1],
      best_acc_valid[num_train_steps-1], best_acc_test[num_train_steps - 1])

# Show graph with all the validation loss curves from training
plt.legend(loc="best")
fig_valid_loss.savefig("2_2_1_LR_tune.png")
plt.show()

# Set up and plot graph showing training & validation accuracy
fig_acc = plt.figure(2)
fig_acc.patch.set_facecolor('white')
axes2 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

acc_train_plot = plt.plot(plot_steps, best_acc_train, plot_lines[0], label="Training")
acc_valid_plot = plt.plot(plot_steps, best_acc_valid, plot_lines[1], label="Validation")

plt.legend(loc="best")
fig_valid_loss.savefig("2_2_1_acc.png")
plt.show()

# Set up and plot graph showing training & validation loss
fig_loss = plt.figure(3)
fig_loss.patch.set_facecolor('white')
axes3 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

loss_train_plot = plt.plot(plot_steps, best_loss_train, plot_lines[0], label="Training")
loss_valid_plot = plt.plot(plot_steps, best_loss_valid, plot_lines[1], label="Validation")

plt.legend(loc="best")
fig_valid_loss.savefig("2_2_1_ce_loss.png")
plt.show()


