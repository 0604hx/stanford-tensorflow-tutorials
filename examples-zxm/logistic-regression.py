"""逻辑回归联系

add on 2017年11月2日14:56:21
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as  tf
import time

# 读取 mnist 数据 （位于根目录下 data/mnist）
# 如果数据文件不存在tensorFlow会尝试从 googleapis 中下载，无奈国内是无法访问到这些资源的 =.=
MNIST = input_data.read_data_sets("../data/mnist", one_hot=True)

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# 定义输入输出
# MNIST 中每张图片是 28*28 像素，则共 784 个点数据，所以输入是 1*784 的1阶张量
# 结果数据是 0-9 ，用 1*10 的 one-hot 向量表示
# 如 0 为 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 如 4 为 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
X = tf.placeholder(tf.float32, [batch_size, 784], name="X_placeholder")
Y = tf.placeholder(tf.int32, [batch_size, 10], name="Y_placeholder")

# 定义权重及偏移量
W = tf.Variable(tf.zeros([784, 10]),name="weights")
b = tf.Variable(tf.zeros([1,10]), name="bias")

# 构建模型
Y_predicted = tf.matmul(X, W) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_predicted, labels=Y, name="loss")
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    start_time = time.time()

    sess.run(tf.global_variables_initializer())

    batchs = int(MNIST.train.num_examples / batch_size)

    for i in range(n_epochs):
        total_loss = 0

        for batchIndex in range(batchs):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})

            total_loss += loss_batch

        print('Average loss epoch {0}: {1}'.format(i, total_loss / batchs))

    print("Train finished! Total time: {} seconds".format(time.time() - start_time))

    # 开始测试模型
    predicted = tf.nn.softmax(Y_predicted)
    correct_predicted = tf.equal(tf.argmax(predicted, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_predicted, tf.int32))

    total_correct_predicted = 0
    test_batchs = int(MNIST.test.num_examples / batch_size)

    for i in range(test_batchs):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        print(Y_batch)
        accuracy_batch = sess.run(accuracy, feed_dict={X:X_batch, Y:Y_batch})
        total_correct_predicted += accuracy_batch

    print("Total accuracy {}, rate {} %".format(total_correct_predicted, 100*total_correct_predicted / MNIST.test.num_examples))
