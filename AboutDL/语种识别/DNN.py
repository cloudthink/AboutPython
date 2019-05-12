import timeit
import numpy as np
import tensorflow as tf
import input_data
import os
from rbm import RBM
from YZSB import FixNN

class DNN(object):
    def __init__(self, n_in, n_out=3, hidden_layers_sizes=[2048, 2048, 50, 2048, 2048]):
        assert len(hidden_layers_sizes) > 0
        self.rbm_layers = []
        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=None)
        self.y = tf.placeholder(tf.float32, shape=None)

        #构筑DBN
        for i in range(len(hidden_layers_sizes)):
            if i == 0:
                layer_input = self.x
                input_size = n_in
            else:
                input_size = hidden_layers_sizes[i-1]
            # 隐层
            bound_val = 4.0*np.sqrt(6.0/(input_size + hidden_layers_sizes[i]))
            W = tf.Variable(tf.random_uniform([input_size, hidden_layers_sizes[i]], minval=-bound_val, maxval=bound_val),dtype=tf.float32, name="W{}".format(i))
            b = tf.Variable(tf.zeros([hidden_layers_sizes[i],]), dtype=tf.float32, name="b{}".format(i))
            #sum_W = tf.matmul(layer_input, W) + b
            sum_W = tf.add(tf.matmul(layer_input, W), b, name="HiddenLayer{}".format(i))
            t_layer_input = tf.nn.sigmoid(sum_W)
            if i > 0 and hidden_layers_sizes[i-1] > hidden_layers_sizes[i]:
                self.DBF = t_layer_input
            # 创建RBM层
            self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],W=W, hbias=b))
            layer_input = t_layer_input

        W = tf.Variable(tf.zeros([hidden_layers_sizes[-1], n_out], dtype=tf.float32))
        b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32)
        self.output = tf.nn.softmax(tf.matmul(layer_input, W) + b)
        self.y_pred = tf.argmax(self.output, axis=1)
        self.loss = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.output), axis=1))#cross_entropy
        correct_pred = tf.equal(self.y_pred, tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #预训练
    def pretrain(self,X_train,sess=None, batch_size=100, pretraining_epochs=10, lr=0.005, k=1, display_step=1):
        if sess is None:
            sess = self.sess
        print('开始预训练...\n')
        start_time = timeit.default_timer()
        batch_num = int(X_train.train.num_examples / batch_size)
        #预训练RBM层
        for i in range(len(self.rbm_layers)):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for _ in range(batch_num):
                    x_batch, _ = X_train.train.next_batch(batch_size)
                    sess.run(train_ops, feed_dict={self.x: x_batch,})
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,}) / batch_num
                if epoch % display_step == 0:
                    print("\t预训练层 {0} 步数 {1} :{2}".format(i+1, epoch+1,avg_cost))
        end_time = timeit.default_timer()
        print("\n预训练进程用时 {0} 分钟".format((end_time - start_time) / 60))
    
    #训练拟合神经网络
    def finetuning(self, trainSet,sess=None,training_epochs=500, batch_size=100, lr=0.1,display_step=1):
        if sess is None:
            sess = self.sess
        print("\nDNN训练...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)
        epoch = 1
        maxAcc = 0
        while epoch <= training_epochs:
            avg_loss = 0.0
            batch_num = int(trainSet.train.num_examples / batch_size)
            for _ in range(batch_num):
                x_batch, y_batch = trainSet.train.next_batch(batch_size)
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                avg_loss += sess.run(self.loss, feed_dict= {self.x: x_batch, self.y: y_batch}) / batch_num
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.validation.wavs,self.y: trainSet.validation.labels})
                maxAcc = val_acc if val_acc > maxAcc else maxAcc
                print("\t训练步数 {0} 校验精度确认: {1} 损失值:{2}".format(epoch, val_acc,avg_loss))
            if epoch == training_epochs:
                conFlag = input('达到最大训练步数，是否增加步数继续训练？Yes/No：')
                if conFlag == 'Yes':
                    training_epochs += int(input('请输入延长的步数（整数）：'))
            epoch += 1
        print('最佳校验精度：{}'.format(maxAcc))
        end_time = timeit.default_timer()
        print("\n训练进程耗时 {0} 分钟".format((end_time - start_time) / 60))
        self.TestAcc(trainSet)

    def TestAcc(self,trainSet):
        val_acc = self.sess.run(self.accuracy, feed_dict={self.x: trainSet.test.wavs,self.y: trainSet.test.labels})
        print("\tDNN测试精度: {0}".format(val_acc))

    def predect(self,x,y=None):
        y_pred = self.sess.run(self.y_pred,feed_dict={self.x:[x]})
        print("预测标签："+input_data.rev_lab_dict[y_pred[0]])
        if y is not None:
            print("真实标签：{}".format(input_data.rev_lab_dict[np.dot(np.array(y),input_data.rev_ten)]))

    def GetDeepFeature(self,X):
        return self.sess.run(self.DBF,feed_dict={self.x:X})

    def load(self, modelName='yzsb',path='D:\\YZSB'):
        folder = os.path.join(path, "{}".format(modelName),"Model.ckpt")
        print("模型加载中...")
        #因为在类的初始化方法中构造了计算过程，所以这里不加载计算图(加载反倒会报错)
        #saver = tf.train.import_meta_graph("{}.meta".format(folder))#不重复定义运算时用这个
        saver = tf.train.Saver()
        #使用restore前需要定义计算图上的所有运算
        saver.restore(self.sess, folder)
        print("加载模型来自 " + folder)
        return self

    def save(self,sess=None, modelName='yzsb', path='D:\\YZSB'):
        if sess is None:
            sess = self.sess
        folder = os.path.join(path, "{}".format(modelName),"Model.ckpt")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        print("保存模型中...")
        saver = tf.train.Saver()
        saver.save(sess, folder)
        print("保存完成")
        

#通过命令行模式启动时执行
if __name__ == "__main__":
    data = input_data.read_data_sets("D:\\DataSet\\")
    dnn = DNN(n_in=input_data.mfcc_length*input_data.frame_length, n_out=3, hidden_layers_sizes=[2048, 2048, 50, 2048, 2048])
    modelName = 'yzsb500'#保存和加载模型的名字
    if os.path.exists(os.path.join('D:\\YZSB',modelName,"Model.ckpt.meta")):
        dnn.load(modelName=modelName)
        dnn.TestAcc(trainSet=data)
        dnn.predect(data.test.wavs[0],data.test.labels[0])
    else:
        init = tf.global_variables_initializer()
        dnn.sess.run(init)
        tf.set_random_seed(seed=2019)
        dnn.pretrain(X_train=data)
        dnn.finetuning(trainSet=data)
        dnn.save(modelName=modelName)

    yzsb = FixNN()
    x1 = dnn.GetDeepFeature(data.train.wavs)
    x2 = dnn.GetDeepFeature(data.validation.wavs)
    x = np.concatenate((x1,x2),axis=0)
    y = np.concatenate((data.train.labels, data.validation.labels),axis=0)
    yzsb.fit(x,y,dnn.GetDeepFeature(data.test.wavs), data.test.labels)
    a = input('训练完成')