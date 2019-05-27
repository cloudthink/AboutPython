import numpy as np
import random
import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,Dropout,Activation#,Flatten,Conv2D,LSTM,MaxPooling2D
from keras.models import Sequential,Model,Input
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

modelName = 'DNN4yzsb.h5'
class FixNN(object):
    def __init__(self,shape = input_data.mfcc_length*input_data.frame_length,learning_rate = 0.0001,batch_size=128,n_epochs = 100,dropout = 0.5):
        self.lab_dict = input_data.lab_dict
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout=dropout
        inputs = Input(shape=(shape,))
        H1 = Dense(1024,activation='relu')(inputs)
        H2 = Dense(512,activation='relu')(inputs)
        H3 = Dense(256,activation='relu')(H2)
        H4 = Dense(len(input_data.lab_dict),activation='relu')(H3)
        H5 = Dense(256,activation='relu')(H4)
        H6 = Dense(512,activation='relu')(H5)
        H7 = Dense(1024,activation='relu')(H6)
        outputs = Dense(shape,activation='sigmoid')(H6)
        self.Classifer = Model(input = inputs,output=H4)
        opt = keras.optimizers.Adam(lr = learning_rate)
        self.Classifer.compile(loss=keras.losses.mean_squared_error,optimizer=opt,metrics=['accuracy'])

    def fit(self,x,y,xt,yt):
        checkpointer = ModelCheckpoint(filepath=modelName, save_best_only=True)
        tensbrd = TensorBoard(log_dir='./tmp/tbLog')
        eStop = EarlyStopping(patience=10)#损失函数不再减小后十轮停止训练
        callback = [checkpointer]
        
        history = self.Classifer.fit(x,y,epochs=self.n_epochs,batch_size=self.batch_size,validation_split=0.2,callbacks=callback)
        self.TestAcc(xt,yt)

        plt.plot(np.arange(len(history.history['acc'])),history.history['acc'],label='Train')
        plt.plot(np.arange(len(history.history['val_acc'])),history.history['val_acc'],label='CV')
        plt.title('Accuracy')
        plt.xlabel('Epcho')
        plt.ylabel('ACC')
        plt.legend(loc=0)
        plt.show()

    def load(self):
        self.Classifer = keras.models.load_model(modelName)

    def TestAcc(self,xt,yt):
        loss,acc = self.Classifer.evaluate(xt,yt)
        print("模型验证集损失值: %f 精确度: %f"%(loss,acc))

    def predict(self,x,y):
        pred = np.argmax(self.Classifer.predict(x), axis=1)
        for i,p in enumerate(pred):
            print("预测标签：{}".format(input_data.rev_lab_dict[p]))
            print("真实标签：{}\n".format(input_data.rev_lab_dict[np.dot(np.array(y[i]),input_data.rev_ten)]))

if __name__ == "__main__":
    data = input_data.read_data_sets("/home/yangjinming/DataSet/")
    nn = FixNN()
    x = np.concatenate((data.train.wavs,data.validation.wavs),axis=0)
    y = np.concatenate((data.train.labels, data.validation.labels),axis=0)
    nn.fit(x,y,data.test.wavs,data.test.labels)#训练时自动保存
    #nn.load()#训练过后直接加载,其实这个模型训练速度极快，实验时每次都训练不用加载也可以
    #nn.TestAcc(data.test.wavs,data.test.labels)
    nn.predict(data.test.wavs[:10],data.test.labels[:10])
    a=input('训练结束')
