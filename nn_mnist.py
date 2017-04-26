import gzip
import cPickle

import tensorflow as tf
import numpy as np


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


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Conjunto de entreno, validacion y test
train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y,10)
valid_y = one_hot(valid_y,10)
test_y = one_hot(test_y,10)



# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt
array = []

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
#se crea el modelo
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels-- respuestas correctas

#variable=tensor

W1 = tf.Variable(np.float32(np.random.rand(28*28,100)) * 0.1) #pesos
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1) #bias

W2 = tf.Variable(np.float32(np.random.rand(100,10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2) #implementacion  asignar prob #Evidencia de que es un num---< probabilidad
#lista valores entre 0-1 +1
#tf.log---calcular log de cada elemento de y
#t.reduce anade los elementos de la segunda  dim de y
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # learning rate: 0.01


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
#entreno
ulterror = 1000
limit = 0.001
array2 = []

for epoch in xrange(10):  #numero epocas
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys =train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: batch_xs})
    #for b, r in zip(batch_ys, result):
        #print b, "-->", r
    print "----------------------------------------------------------------------------------"

    error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    #array.append(error)
    if abs(error - ulterror) < limit:
        break

    ulterror = error
    #Test del entreno

    # etiquetaprobable vs etiqueta real
    # dev lista booleanos
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # se pasa la lista a num y se halla la media
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    final = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print(final)
    array2.append((final,error))




plt.plot(array2)
plt.show()