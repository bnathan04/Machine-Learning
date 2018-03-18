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

# Set loss function parameters
learning_rate = [0.005, 0.001, 0.0001]
weight_decay_coeff = 0

# Prepare for training runs
num_train_steps = 20000
batch_size = 500
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_batches
loss_train = np.zeros(num_train_steps)

# Set up the loss calculations
prediction = tf.matmul(X, W) + b
mse = tf.reduce_mean(tf.square(prediction - Y)) / 2  # needs to be modified to encapsulate decay weight later on

# Graph set up
plot_lines = ["b-", "r-", "g-"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)
fig = plt.figure()
fig.patch.set_facecolor('white')
axes = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Train
for count, rate in enumerate(learning_rate):

    # Start tf session, init variables and SGD optimizer function
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(mse)

    # Run training
    for step in range(num_train_steps):
        cur_batch_idx = (step % num_batches) * batch_size
        cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
        cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]
        sess.run(optimizer, feed_dict={X: cur_data, Y: cur_target})

        loss_train[step] = sess.run(mse, feed_dict={X: cur_data, Y: cur_target})

        # if step % batch_size == 0:
        #     print(rate, step, loss_train[step])

    # Plot current run
    print(rate, loss_train[num_train_steps-1])
    plot_name = "Learning Rate: " + str(rate)
    loss_plot = plt.plot(plot_steps, loss_train, plot_lines[count], label=plot_name)

# Save graph
plt.legend(loc="best")
fig.savefig("1_1_loss_LR.png")
plt.show()

# print("Done")







