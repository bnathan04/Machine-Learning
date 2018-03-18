import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import copy


def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                     data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
                                     data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                           target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
                                           target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Cross Entropy Loss calculation function
def calculate_ce_loss (W, truth, prediction, coeff):

    regularizer = (coeff / 2) * tf.reduce_sum(tf.square(W))
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=prediction))
    total_loss = ce_loss + regularizer
    return total_loss


# Acquire data
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0)

# Generate random values for the W matrix and bias term from a normal distribution
W = tf.Variable(tf.truncated_normal(shape=[trainData.shape[1], 6], stddev=0.1, dtype=tf.float64))
b = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, dtype=tf.float64))

# Set up place holders for the tf graph
X_train = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Train_Data")
Y_train = tf.placeholder(tf.float64, shape=[None], name="Train_Label")
X_valid = tf.placeholder(tf.float64, shape=[None, trainData.shape[1]], name="Valid_Data")
Y_valid = tf.placeholder(tf.float64, shape=[None], name="Valid_Label")
Lambda = tf.placeholder(tf.float64, shape=[1, ], name="Weight_Decay")
data = tf.placeholder(tf.float64, name="Data")
labels = tf.placeholder(tf.int32, shape=[None], name="Labels")

# Set up parameters, including learning rate/weight decay to be tuned
# Set empty arrays to record values at each step of training
weight_decay = np.array([0.0, 0.001, 0.1, 1])
weight_decay = np.reshape(weight_decay, (4, -1))

learning_rate = [0.005, 0.001, 0.0001]
batch_size = 300
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
best_weight = 0

# Loss calculations
prediction_train = tf.matmul(X_train, W) + b
prediction_valid = tf.matmul(X_valid, W) + b
ce_loss_train = calculate_ce_loss(W, tf.one_hot(tf.cast(Y_train, tf.int32), depth=6, axis=-1),
                                  prediction_train, Lambda)
ce_loss_valid = calculate_ce_loss(W, tf.one_hot(tf.cast(Y_valid, tf.int32), depth=6, axis=-1),
                                  prediction_valid, Lambda)

# Classification
y_hat = tf.nn.softmax(tf.matmul(data, W) + b)
is_correct = tf.equal(tf.argmax(y_hat, 1), tf.cast(labels, tf.int64))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Graph set up
plot_lines = ["b-", "r-", "-g"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)
values = range(12)
cmap = plt.get_cmap('gnuplot')
plot_colors = [cmap(i) for i in np.linspace(0, 1, 12)]
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

fig_valid_loss = plt.figure(1)
fig_valid_loss.patch.set_facecolor('white')
axes1 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

sess = tf.Session()
plot_count = 0

# Train -- iterate across all possible weight decay and learning rate combinations to tune
# Use validation loss as tuning metric
for weight_count, weight in enumerate(weight_decay):
    for count, rate in enumerate(learning_rate):

        # Start session, (re)init variables and optimizer
        sess = tf.Session()
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(ce_loss_train)
        sess.run(tf.global_variables_initializer())

        # Start training loop
        for train_step in range(num_train_steps):

            # Get current batch data
            cur_batch_idx = (train_step % num_batches) * batch_size
            cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
            cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]

            # Optimize and get the current loss calculations
            [optimizer_value, loss_train[train_step], loss_valid[train_step]] = \
                sess.run(fetches=[optimizer, ce_loss_train, ce_loss_valid],
                         feed_dict={X_train: cur_data, Y_train: cur_target,
                                    X_valid: validData, Y_valid: validTarget, Lambda: weight})

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
        if count == 0 and weight_count == 0:
            best_rate = count
            best_weight = weight_count
            best_loss_valid = copy.deepcopy(loss_valid)
            best_acc_valid = copy.deepcopy(acc_valid)
            best_loss_train = copy.deepcopy(loss_train)
            best_acc_train = copy.deepcopy(acc_train)
            best_acc_test = copy.deepcopy(acc_test)

        if best_loss_valid[num_train_steps - 1] > loss_valid[num_train_steps - 1]:
            best_rate = count
            best_weight = weight_count
            best_loss_valid = copy.deepcopy(loss_valid)
            best_acc_valid = copy.deepcopy(acc_valid)
            best_loss_train = copy.deepcopy(loss_train)
            best_acc_train = copy.deepcopy(acc_train)
            best_acc_test = copy.deepcopy(acc_test)

        # Plot a validation loss curve for each tuning run
        plot_name = "Learning Rate:" + str(rate) + ", Weight Decay:" + str(weight)
        colorVal = scalarMap.to_rgba(values[plot_count])
        loss_valid_plot = plt.plot(plot_steps, loss_valid, "-", color=colorVal, label=plot_name)
        plot_count += 1

# Print out summary of training results
print("Best Learning Rate = ", learning_rate[best_rate])
print("Best Weight Decay Coefficient = ", weight_decay[best_weight])
print("Soft-Max Accuracy (training/validation/test):", best_acc_train[num_train_steps-1],
      best_acc_valid[num_train_steps-1], best_acc_test[num_train_steps - 1])

# Show a graph with the validation loss curves for all the tuning runs
plt.legend(loc="best")
fig_valid_loss.savefig("2_2_2_LR_Weight_tune.png")
plt.show()

# Set up and show a graph with training & validation accuracy at the tuned values
fig_acc = plt.figure(2)
fig_acc.patch.set_facecolor('white')
axes2 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

acc_train_plot = plt.plot(plot_steps, best_acc_train, plot_lines[0], label="Training")
acc_valid_plot = plt.plot(plot_steps, best_acc_valid, plot_lines[1], label="Validation")

plt.legend(loc="best")
fig_acc.savefig("2_2_2_acc.png")
plt.show()

# Set up and show a graph with training & validation loss at the tuned values
fig_loss = plt.figure(3)
fig_loss.patch.set_facecolor('white')
axes3 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

loss_train_plot = plt.plot(plot_steps, best_loss_train, plot_lines[0], label="Training")
loss_valid_plot = plt.plot(plot_steps, best_loss_valid, plot_lines[1], label="Validation")

plt.legend(loc="best")
fig_loss.savefig("2_2_2_ce_loss.png")
plt.show()


