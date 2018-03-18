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


# Cross Entropy Loss calculation function
def calculate_ce_loss (W, truth, prediction, coeff):

    regularizer = (coeff / 2) * tf.reduce_sum(tf.square(W))
    ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=prediction))
    total_loss = ce_loss + regularizer
    return total_loss


# Generate random values for the W matrix and bias term from a normal distribution
W = tf.Variable(tf.truncated_normal(shape=[trainData.shape[1], 1], stddev=0.1, dtype=tf.float64))
b = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, dtype=tf.float64))

# Set up place holders for the tf graph
X_train = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Train_Data")
Y_train = tf.placeholder(tf.float64, shape=[None, 1], name="Train_Label")
X_valid = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Valid_Data")
Y_valid = tf.placeholder(tf.float64, shape=[None, 1], name="Valid_Label")
X_test = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Test_Data")
Y_test = tf.placeholder(tf.float64, shape=[None, 1], name="Test_Label")
data = tf.placeholder(tf.float64, name="Data")
labels = tf.placeholder(tf.float64, name="Labels")

# Set up parameters, including learning rate to be tuned, and set empty arrays to record values at each step of training
weight_decay = 0.01
learning_rate = [0.005, 0.001, 0.0001]
batch_size = 500
num_train_steps = 5000
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_data
loss_train = np.zeros(num_train_steps)
acc_train = np.zeros(num_train_steps)
loss_test = np.zeros(num_train_steps)
acc_test = np.zeros(num_train_steps)
loss_valid = np.zeros(num_train_steps)
acc_valid = np.zeros(num_train_steps)
best_acc_train = []
best_loss_train = []
best_acc_test = []
best_loss_test = []
best_acc_valid = []
best_loss_valid = np.full((num_train_steps), 99)
best_rate = 0

# Loss and accuracy calculations ----

# Cross Entropy
ce_loss_train = calculate_ce_loss(W, Y_train, tf.matmul(X_train, W) + b, weight_decay)
ce_loss_test = calculate_ce_loss(W, Y_test, tf.matmul(X_test, W) + b, weight_decay)
ce_loss_valid = calculate_ce_loss(W, Y_valid, tf.matmul(X_valid, W) + b, weight_decay)

# Classification with logistic regression
class_prediction = tf.sigmoid(tf.matmul(data, W) + b)
classify = tf.cast(tf.greater(class_prediction, 0.5), tf.float64)
is_correct = tf.reduce_sum(tf.cast(tf.equal(classify, tf.cast(labels, tf.float64)), tf.float64))
accuracy = tf.cast(is_correct, tf.float64) / tf.cast(tf.shape(classify)[0], tf.float64)

# Train
for count, rate in enumerate(learning_rate):

    # Start session, (re)init variables and optimizer
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(ce_loss_train)

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
        if train_step % batch_size == 0:
            print("Train loss:", loss_train[train_step], "Valid loss:", loss_valid[train_step],
                  "Train accuracy:", acc_train[train_step], "Valid accuracy:", acc_valid[train_step],
                  "Test accuracy:", acc_test[train_step])

    # Choose best learning rate based using validation cross entropy loss as metric
    if count == 0:
        best_rate = count
        best_loss_valid = copy.deepcopy(loss_valid)
        best_acc_train = copy.deepcopy(acc_train)
        best_loss_train = copy.deepcopy(loss_train)
        best_acc_test = copy.deepcopy(acc_test)
        best_acc_valid = copy.deepcopy(acc_valid)

    elif best_loss_valid[num_train_steps - 1] > loss_valid[num_train_steps - 1]:
        best_rate = count
        best_loss_valid = copy.deepcopy(loss_valid)
        best_acc_train = copy.deepcopy(acc_train)
        best_loss_train = copy.deepcopy(loss_train)
        best_acc_test = copy.deepcopy(acc_test)
        best_acc_valid = copy.deepcopy(acc_valid)


print("Best Learning Rate = ", learning_rate[best_rate])
print("Best Classification Accuracy (train/valid/test) = ", best_acc_train[num_train_steps - 1],
      best_acc_valid[num_train_steps - 1], best_acc_test[num_train_steps - 1])

# Graphs set up
plot_lines = ["b-", "r-"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)

fig_loss = plt.figure(1)
fig_loss.patch.set_facecolor('white')
axes1 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Loss")

loss_plot_train = plt.plot(plot_steps, best_loss_train, plot_lines[0], label="Training Loss")
loss_plot_valid = plt.plot(plot_steps, best_loss_valid, plot_lines[1], label="Validation Loss")

plt.legend(loc="best")
fig_loss.savefig("2_1_1_loss.png")
plt.show()


fig_acc = plt.figure(2)
fig_acc.patch.set_facecolor('white')
axes2 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

acc_plot_train = plt.plot(plot_steps, best_acc_train, plot_lines[0], label="Training Accuracy")
acc_plot_valid = plt.plot(plot_steps, best_acc_valid, plot_lines[1], label="Validation Accuracy")

plt.legend(loc="best")
fig_acc.savefig("2_1_1_accuracy.png")
plt.show()

print("We got here safely")
