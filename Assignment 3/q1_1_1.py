import tensorflow as tf
import numpy as np 

# with np.load("notMNIST.npz") as data:
#     Data, Target = data["images"], data["labels"]
#     np.random.seed(521)
#     randIndx = np.arange(len(Data))
#     np.random.shuffle(randIndx)
#     Data = Data[randIndx]/255.
#     Target = Target[randIndx]
#     trainData, trainTarget = Data[:15000], Target[:15000]
#     validData, validTarget = Data[15000:16000], Target[15000:16000]
#     testData, testTarget = Data[16000:], Target[16000:]

# print "Train Data Shape:", trainData.shape
# print "Train Target Shape:", trainTarget.shape
# print "Valid Data Shape:", validData.shape
# print "Valid Target Shape:", validTarget.shape     
# print "Test Data Shape:", testData.shape 
# print "Test Target Shape:", testTarget.shape 

# trainData = np.reshape(trainData, (trainData.shape[0], -1))
# validData = np.reshape(validData, (validData.shape[0], -1))
# testData = np.reshape(testData, (testData.shape[0], -1))

# print "-------- New shapes ----------"
# print "Train Data Shape:", trainData.shape
# print "Train Target Shape:", trainTarget.shape
# print "Valid Data Shape:", validData.shape
# print "Valid Target Shape:", validTarget.shape
# print "Test Data Shape:", testData.shape
# print "Test Target Shape:", testTarget.shape

def build_layer(X, num_hidden_units):
    
    # initialize the weight matrix and bias vector
    num_inputs = X.get_shape().as_list()[0]
    W = tf.get_variable(name="Weights", shape=(num_inputs, num_hidden_units), dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(tf.zeros(shape=(num_hidden_units, 1), dtype=tf.float64), name="Bias")

    # input to next layer
    z = tf.add(tf.matmul(tf.transpose(W), X), b)

    return z

