import timeit
import numpy as np
import tensorflow as tf
import input_data
import os
from mlp import HiddenLayer,LogisticRegression
from rbm import RBM
from YZSB import FixNN

class DNN(object):
    def __init__(self, n_in=43, n_out=3, hidden_layers_sizes=[2048, 2048, 43, 2048, 2048]):
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []
        self.rbm_layers = []
        self.params = []

        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=None)
        self.y = tf.placeholder(tf.float32, shape=None)

        #构筑DBN
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                input_size = n_in
            else:
                layer_input = self.layers[i-1].output
                input_size = hidden_layers_sizes[i-1]
            # Sigmoid层
            sigmoid_layer = HiddenLayer(inpt=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],activation=tf.nn.sigmoid)
            self.layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            # 创建RBM层
            self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],W=sigmoid_layer.W, hbias=sigmoid_layer.b))

        self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1], n_out=n_out)
        self.params.extend(self.output_layer.params)
        #for p in self.params:
            #tf.add_to_collection(p.name,p)
        self.cost = self.output_layer.cost(self.y)
        self.accuracy = self.output_layer.accuarcy(self.y)

    #预训练
    def pretrain(self,X_train,sess=None, batch_size=100, pretraining_epochs=10, lr=0.005, k=1, display_step=1):
        if sess is None:
            sess = self.sess
        print('开始预训练...\n')
        start_time = timeit.default_timer()
        batch_num = int(X_train.train.num_examples / batch_size)
        #预训练RBM层
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    x_batch, _ = X_train.train.next_batch(batch_size)
                    sess.run(train_ops, feed_dict={self.x: x_batch,})
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,}) / batch_num
                if epoch % display_step == 0:
                    print("\t预训练层 {0} 步数 {1} :{2}".format(i+1, epoch+1,avg_cost))

        end_time = timeit.default_timer()
        print("\n预训练进程用时 {0} 分钟".format((end_time - start_time) / 60))
    
    #训练拟合神经网络
    def finetuning(self, trainSet,sess=None,training_epochs=300, batch_size=100, lr=0.1,display_step=1):
        if sess is None:
            sess = self.sess
        print("\nDNN训练...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.cost)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(trainSet.train.num_examples / batch_size)
            for i in range(batch_num):
                x_batch, y_batch = trainSet.train.next_batch(batch_size)
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                avg_cost += sess.run(self.cost, feed_dict= {self.x: x_batch, self.y: y_batch}) / batch_num
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.validation.wavs,self.y: trainSet.validation.labels})
                print("\t训练步数 {0} 校验精度确认: {1} 损失值:{2}".format(epoch+1, val_acc,avg_cost))

        val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.test.wavs,self.y: trainSet.test.labels})
        print("\tDNN测试精度: {0}".format(val_acc))
        end_time = timeit.default_timer()
        print("\n训练进程耗时 {0} 分钟".format((end_time - start_time) / 60))

    def GetDeepFeature(self,X,training_epochs=10):
        x = tf.placeholder(tf.float32, shape=None)
        temp = tf.nn.relu(tf.matmul(x, self.params[0]) + self.params[1])
        for i in range(1,3):
            temp = tf.nn.relu(tf.matmul(temp, self.params[2*i]) + self.params[2*i+1])
        return self.sess.run(temp,feed_dict={x:X})

    def load(self, run_id='yzsb',path='D:\\YZSB'):
        folder = os.path.join(path, "{}".format(run_id),"Model.ckpt")
        print("模型加载中...")
        saver = tf.train.import_meta_graph("{}.meta".format(folder))
        saver.restore(self.sess, folder)
        print("加载模型来自 " + folder)
        return self

    def save(self,sess=None, run_id='yzsb', path='D:\\YZSB'):
        if sess is None:
            sess = self.sess
        folder = os.path.join(path, "{}".format(run_id),"Model.ckpt")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        print("保存模型中...")
        saver = tf.train.Saver()
        saver.save(sess, folder)
        print("保存完成")
        

#通过命令行模式启动时执行
if __name__ == "__main__":
    data = input_data.read_data_sets("D:\\DataSet\\", one_hot=True)
    dnn = DNN(n_in=input_data.mfcc_length*input_data.frame_length, n_out=3, hidden_layers_sizes=[2048, 2048, 50, 2048, 2048])
    if os.path.exists(os.path.join('D:\\YZSB','yzsb',"Model.ckpt.meta")):
        dnn.load()
        sess = dnn.sess
        init = tf.global_variables_initializer()
        dnn.sess.run(init)
    else:
        init = tf.global_variables_initializer()
        dnn.sess.run(init)
        tf.set_random_seed(seed=2019)
        dnn.pretrain(X_train=data)
        dnn.finetuning(trainSet=data)
        dnn.save()

    yzsb = FixNN()
    x = np.concatenate(dnn.GetDeepFeature(data.train.wavs),dnn.GetDeepFeature(data.validation.wavs))
    y = np.concatenate(data.train.labels, data.validation.labels)
    yzsb.fit(x,y,dnn.GetDeepFeature(data.test.wavs), data.test.labels)
    a=input('训练完成')
