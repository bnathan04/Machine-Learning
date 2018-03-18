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

# Set up parameters, including learning rate to be tuned, and set empty arrays to record values at each step of training
weight_decay = 0.01
learning_rate = 0.001
batch_size = 500
num_train_steps = 5000
num_data = trainData.shape[0]
num_batches = num_data // batch_size
num_epoch = num_train_steps / num_data
loss_train = np.zeros(num_train_steps)

# Loss calculations ----

# Cross Entropy
ce_loss_train = calculate_ce_loss(W, Y_train, tf.matmul(X_train, W) + b, weight_decay)

# Set up different optimizers
optimizers = [tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(ce_loss_train),
              tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ce_loss_train)]

# Graph set up
plot_lines = ["b-", "r-"]
plot_steps = np.linspace(0, num_epoch, num=num_train_steps)

fig_loss = plt.figure(1)
fig_loss.patch.set_facecolor('white')
axes1 = plt.gca()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plot_names = ["SGD", "Adam"]

# Train
for count, algo in enumerate(optimizers):

    # Start session, (re)init variables and optimizer
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    optimizer = algo

    for train_step in range(num_train_steps):

        # Get current batch data
        cur_batch_idx = (train_step % num_batches) * batch_size
        cur_data = trainData[cur_batch_idx:cur_batch_idx + batch_size]
        cur_target = trainTarget[cur_batch_idx:cur_batch_idx + batch_size]

        # Optimize and get the current loss calculations
        [optimizer_value, loss_train[train_step]] = sess.run(fetches=[optimizer, ce_loss_train],
                                                             feed_dict={X_train: cur_data, Y_train: cur_target})

        # Every batch, print out a summary
        if train_step % batch_size == 0:
            print("Train loss:", loss_train[train_step])

    # Plot
    loss_plot_train = plt.plot(plot_steps, loss_train, plot_lines[count], label=plot_names[count])

plt.legend(loc="best")
fig_loss.savefig("2_1_2_SGD_ADAM.png")
plt.show()

print("We got here safely")







