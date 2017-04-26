import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
array = []
array2 = []
# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x_train = x_data[0:(int)(0.7*len(x_data)), :]
y_train = y_data[0:(int)(0.7*len(x_data)), :]

x_valid = x_data[(int)(0.7*len(x_data)):(int)(0.85*len(x_data)), :]
y_valid = y_data[(int)(0.7*len(x_data)):(int)(0.85*len(x_data)), :]

x_test = x_data[(int)(0.85*len(x_data)):, :]
y_test = y_data[(int)(0.85*len(x_data)):, :]

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
#loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
ulterror = 100
limit = 0.000000000000001
epoca = 100


for epoch in xrange(epoca):


    for jj in xrange(len(x_train) / batch_size):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: batch_xs})
    #for b, r in zip(batch_ys, result):
      #  print b, "-->", r
    error = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})
    #array.append(error)

    if abs(error - ulterror) < limit:
        break

    ulterror = error

    # Porcentaje aciertos
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # se pasa la lista a num y se halla la media
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    final=sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    print(final)
    array2.append((final,error))

    print "----------------------------------------------------------------------------------"

#Porcentaje aciertos
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
#se pasa la lista a num y se halla la media
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

plt.plot(array2)
plt.show()