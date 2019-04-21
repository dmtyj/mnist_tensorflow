from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("", one_hot=True)  # 添加数据源地址


# 参数w初始化
def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 参数b初始化
def bias_varible(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层计算
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层计算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder()  # 设置输入占位符
y_ = tf.placeholder()  # 设置标签占位符
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_varible()  # 设置第一层卷积层参数w
b_conv1 = bias_varible()  # 设置第一层卷积层参数b
h_conv1 = tf.nn.relu()  # 第一层卷积层
h_pool1 = max_pool_2x2()   # 第一层池化层

W_conv2 = weight_varible([5, 5, 32, 64])  # 设置第二层卷积层参数w
b_conv2 = bias_varible([64])  # 设置第二层卷积层参数b
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二层卷积层
h_pool2 = max_pool_2x2(h_conv2)  # 第二层池化层

W_fc3 = weight_varible([7 * 7 * 64, 1024])  # 设置第一层全连接层参数w
b_fc1 = bias_varible([1024])  # 设置第一层全连接层参数b
h_pool2_flat = tf.reshape()  # 对第二层卷积层结果维度转换
h_hc3 = tf.nn.relu()  # 第一层全连接层计算

keep_prob = tf.placeholder(tf.float32)
h_fc3_drop = tf.nn.dropout()  # 第一层全连接层Dropout

W_fc4 = weight_varible()  # 设置第二层全连接层参数w
b_fc4 = bias_varible()  # 设置第二层全连接层参数b
y = tf.add()  # 输出层计算

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
sess.close()
