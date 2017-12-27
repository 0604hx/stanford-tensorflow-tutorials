"""
简单神经网络模型的使用

目的： 求解 A * X * X + B 中的 A、B

训练数据    通过 numpy 随机生成的 200 个 -100 到 100 的均匀分布值

add on 2017年11月9日09:57:13
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义训练数据
data_len = 200
x_data  = np.linspace(-0.8, 0.8, data_len)[:, np.newaxis] # 得到 shape = [200,1] 的数据集
noise   = np.random.normal(0,0.02,x_data.shape)           # 制造一些噪音
y_data  = np.square(x_data) + noise                     # 得到真实的结果集

# 开始定义我们的第一个神经网络，激动！
# 整个神经网络（Neural Networks）共有三层：
#   输入层：这里是 x_data 中的一个元素，如 [-10]
#   中间层：此处是给输入层进行权重划分，设定有10个神经元
#   输出层：这里是经过我们神经网络计算后得到的预测值，需要与 y_data 进行比较然后得到损失值继续优化

# 输入层， 列数应跟 训练数据一致
x   = tf.placeholder(tf.float32, [None, 1])
y   = tf.placeholder(tf.float32, [None,1])

# 中间层神经网络
weight_L1   = tf.Variable(tf.random_normal([1,10]))     # 因为要与 x 相乘，所以行数为1， 列数即为神经元的个数
bias_L1     = tf.Variable(tf.zeros([1,10]))             # shape 与 weight 一致
op_L1       = tf.matmul(x, weight_L1) + bias_L1         # 定义本层神经层的输出
result_L1   = tf.nn.tanh(op_L1)

# 输出层
weight_L2   = tf.Variable(tf.random_normal([10,1]))
bias_L2     = tf.Variable(tf.zeros([1,1]))
op_L2       = tf.matmul(result_L1, weight_L2) + bias_L2
predicted   = tf.nn.tanh(op_L2)

# 定义损失函数及优化器
loss    = tf.reduce_mean(tf.square(y - predicted))
train   = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 启动 session 来训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_loss = 0.

    # 训练1000次
    # 当训练 1千 此后，损失值为 loss=0.002161182463169098 mean=0.01628710670056162
    # 当训练 20W 次后， 损失值为 loss=0.0003901582967955619 mean=0.0005201796614023522
    # 当训练 100W 此后， 损失值为 loss=0.00037945318035781384 mean=0.0004106674489856202
    for index in range(1001):
        _, train_loss = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
        total_loss += train_loss
        if index % 10 == 0:
            print("#{:4} training : loss={:<25} mean={:<25}".format(index, train_loss, total_loss/(index+1)))

    # 使用我们的模型得到预测值
    prediction_value = sess.run(predicted, feed_dict={x:x_data})

    # 看图说话
    plt.scatter(x_data, y_data,label='Real data')
    plt.plot(x_data, prediction_value, 'r-',lw=5,label='Predicted data')
    plt.legend()
    plt.show()