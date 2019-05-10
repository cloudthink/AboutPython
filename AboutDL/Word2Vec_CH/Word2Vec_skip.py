import tensorflow as tf
import numpy as np
import math
import vocabulary

max_steps = 100000  # 训练最大迭代次数10w次
batch_size = 128
embedding_size = 128  # 嵌入向量的尺寸
skip_distance = 1  # 相邻汉字数
num_of_samples = 2  # 对每个汉字生成多少样本

vocabulary_size = vocabulary.vocabulary_size  # 字典大小

# numpy中choice()函数的函数原型为choice(a,size,replace,p)
# choice()函数用于在a给出的范围内抽取size个大小的数组成一个一维数组
# 当设置了replace=False则表示组成的这个一维数组中不能够有重复的数字
valid_examples = np.random.choice(100, 16, replace=False)

num_sampled = 64  # 训练时用来做负样本的噪声汉字的数量

with tf.Graph().as_default():
    # train_inputs和train_labels是训练数据及其label的placeholder
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # embeddings是所有5000高频汉字的词向量，向量的维度是128，数值是由
    # random_uniform()函数生成的在-1.0到1.0之间平均分布的数值
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    # embedding_lookup()函数用于选取一个张量里面索引对应的元素，函数原型是：
    # embedding_lookup(params,ids,partition_strategy,name,validate_indices,max_norm)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 用truncated_normal()函数产生标准差为1.0/math.sqrt(embedding_size)的正态分布数据
    # 产生的nce_weights作为NCE loss中的权重参数
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
    # 产生的nce_biases作为NCE loss中的偏置参数
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # 计算词向量embedding在训练数据上的loss
    # nce_loss()函数原型nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,num_true=1,
    # sampled_values,remove_accidental_hits,partition_strategy,name)
    nec_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,labels=train_labels, inputs=embed,
                              num_sampled=num_sampled,num_classes=vocabulary_size)

    # 求nce_loss的均值
    loss = tf.reduce_mean(nec_loss)

    # 创建优化器，学习率为固定的1.0，最小化loss
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # square()函数用于求平方，之后使用reduce_sum()函数求和；keep_dims=True表示求和之后维度不会发生改变
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    normalized_embeddings = embeddings / norm

    # 在标准化后的所有汉字的词向量值中寻找随机抽取的16个汉字对应的词向量值
    # 在这之前，valid_inputs是由数组valid_examples进行constant操作转化为张量得来，
    valid_inputs = tf.constant(valid_examples, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_inputs)

    # 使用matmul()矩阵运算函数计算相似度，函数原型matmul(a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, name)
    # 在函数matmul()的定义中，name参数默认为None，除a和b外其他参数都有默认的False值，在这里我们设参数transpose_b设True，表示对参数b传入的矩阵进行转置
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # 开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 总损失与平均损失
        total_loss = 0
        average_loss = 0

        for step in range(max_steps + 1):
            # 调用generate_batch()函数生成用于训练的batch及其labels，
            batch_inputs, batch_labels = vocabulary.generate_batch(batch_size, num_of_samples, skip_distance)
            if batch_inputs is None:#数据索引到尽头了
                break
            # 运行loss的计算及最小化loss的优化器
            loss_val, _ = sess.run([loss, optimizer], feed_dict={train_inputs: batch_inputs,train_labels: batch_labels})
            # total_loss用于计算总损失，在每一轮迭代后都会与loss_val相加
            total_loss += loss_val

            # 每进行1000轮迭代就输出平均损失的值，并将average_loss和total_loss
            # 重新归零，方便下一个1000轮的计算
            if step > 0 and step % 1000 == 0:
                average_loss = total_loss / 1000
                print("所处训练步数： %d 平均损失值：%f " % (step, average_loss))
                average_loss = 0
                total_loss = 0

            # 每隔5000轮就打印一次与验证汉字最相似的8个汉字
            if step > 0 and step % 5000 == 0:
                # 执行计算相似性的操作
                similar = similarity.eval()
                # 外层循环16次，
                for i in range(16):
                    # 每执行一次最外层的循环，都会得到一个验证汉字对应的nearest，
                    # 这里有8个数据，是与验证汉字最相近的汉字的编号，通过reverse_dictionary可以得到确切的单词
                    nearest = (-similar[i, :]).argsort()[1:8 + 1]

                    # 定义需要打印的字符串，其中valid_word是通过reverse_dictionary得到的验证单词
                    valid_word = vocabulary.reverse_dictionary[valid_examples[i]]
                    nearest_information = "最接近【 %s 】字的是:" % valid_word

                    for j in range(8):
                        # 在8个循环内通过reverse_dictionary得到与验证汉字相近的8个汉字的原型，并改进需要打印的字符串
                        close_word = vocabulary.reverse_dictionary[nearest[j]]
                        nearest_information = " %s %s" % (nearest_information,close_word)
                    # 打印出验证汉字及与验证汉字相近的8个汉字
                    print("验证汉字是: %s"% valid_word)
                    print(nearest_information)
        final_embeddings = normalized_embeddings.eval()