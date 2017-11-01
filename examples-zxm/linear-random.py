import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: 构造虚拟数据:随机生成 100 条数据
data_len = 100
x_input = np.linspace(-2,2, data_len)
y_input = x_input * 3 + np.random.rand(x_input.shape[0]) * 1
data = np.column_stack((x_input, y_input))

# 假设 Y = W*X + b
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
W = tf.Variable(0., name="weight")
U = tf.Variable(0., name="weight2")
b = tf.Variable(0., name="bias")

Y_predicted = X * W  + b

loss = tf.square(Y - Y_predicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        total_loss = 0
        for x,y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss/data_len))
    
    W,U, b = sess.run([W,U,b])
    print("最终计算 w={}, u={}, b={}".format(W,U, b))

# plot the results
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

X, Y = data.T[0], data.T[1]
plt.xlabel(u"火宅次数（每1000户）")
plt.ylabel(u"盗窃次数（每1000人）")
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * W + b, 'ro', label='Predicted data')
plt.legend()
plt.show()