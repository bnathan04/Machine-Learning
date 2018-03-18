import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
Lambda = tf.placeholder(tf.float64, shape=[1, ], name="Weight_Decay")

# Set loss function parameters
learning_rate = 0.005
weight_decay_coeff = np.array([0.0, 0.001, 0.1, 1])
weight_decay_coeff = np.reshape(weight_decay_coeff, (4, -1))

# Prepare for training runs
num_train_steps = 20000
batch_size = 500
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_batches

loss_train = np.zeros(num_train_steps)
acc_valid = np.zeros(num_train_steps)
acc_test = np.zeros(num_train_steps)

# Set up the loss calculations
prediction = tf.matmul(X, W) + b
scale_factor = tf.constant(0.5, dtype=tf.float64)
# regularizer = tf.multiply(tf.multiply(tf.reduce_sum(tf.square(W)), Lambda), scale_factor)
regularizer = (Lambda / 2) * tf.reduce_sum(tf.square(W))
mse = tf.reduce_mean(tf.square(prediction - Y)) / 2 + regularizer
classify = tf.cast(tf.greater(prediction, 0.5), tf.float64)
is_correct = tf.reduce_sum(tf.cast(tf.equal(classify, tf.cast(Y, tf.float64)), tf.float64))
accuracy = tf.cast(is_correct, tf.float64) / tf.cast(tf.shape(classify)[0], tf.float64)

# Graph set up
plot_lines = ["b-", "r-", "g-", "y-"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)
fig = plt.figure()
fig.patch.set_facecolor('white')
axes = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")


# Train
for count, weight in enumerate(weight_decay_coeff):

    # Start tf session, init variables and SGD optimizer function
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

    # Run training
    for step in range(num_train_steps):
        cur_batch_idx = (step % num_batches) * batch_size
        cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
        cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]
        sess.run(optimizer, feed_dict={X: cur_data, Y: cur_target, Lambda: weight})

        loss_train[step] = sess.run(mse, feed_dict={X: cur_data, Y: cur_target, Lambda: weight})
        acc_valid[step] = sess.run(accuracy, feed_dict={X: validData, Y: validTarget})
        acc_test[step] = sess.run(accuracy, feed_dict={X: testData, Y: testTarget})
        # if step % batch_size == 0:
        #     print(rate, step, loss_train[step])

    # Plot current run
    plot_name = "Weight Decay Coefficient: " + str(weight)
    loss_plot = plt.plot(plot_steps, acc_valid, plot_lines[count], label=plot_name)
    final_valid_acc = acc_valid[num_train_steps - 1]
    final_test_acc = acc_test[num_train_steps - 1]
    print(plot_name, "- final validation accuracy =", final_valid_acc, " corresponding test accuracy =", final_test_acc)

# Save graph
plt.legend(loc="best")
fig.savefig("1_3_validation_accuracy_lambda.png")
plt.show()

# print("Done")







