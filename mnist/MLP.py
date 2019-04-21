from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("", one_hot=True)  # 添加数据源地址

in_units = 784
h1_units = 300

w1 = tf.Variable()  # 补全参数初始化
b1 = tf.Variable()  # 补全参数初始化
w2 = tf.Variable()  # 补全参数初始化
b2 = tf.Variable()  # 补全参数初始化

x = tf.placeholder()  # 设置输入占位符
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu()  # 隐含层
hidden_drop1 = tf.nn.dropout(hidden1, keep_prob)  # 隐含层Dropout

y = tf.nn.softmax()   # 输出层
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean()  # 损失函数

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run([y, train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
