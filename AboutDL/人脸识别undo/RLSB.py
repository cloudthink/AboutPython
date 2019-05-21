import numpy as np
import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.layers import LSTM,Dense,Dropout,Flatten,Conv2D,Activation,MaxPooling2D
#from keras.utils import np_utils
from keras.models import Sequential,Model,Input
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

modelName = 'RLSB.h5'
class FixNN(object):
    def __init__(self,shape,learning_rate = 0.01,batch_size=128,n_epochs = 1000,dropout = 0.5):
        self.lab_dict = input_data.lab_dict
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout

        model = Sequential()
        model.add(Conv2D(50, (3, 3), padding='same', input_shape=shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(60, (3, 3), padding='same', input_shape=shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(70, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense((len(self.lab_dict)),activation = 'relu'))

        self.Classifer = model
        opt = keras.optimizers.Adam(lr=learning_rate)
        self.Classifer.compile(loss=keras.losses.mean_squared_error,optimizer=opt,metrics=['accuracy'])

    def fit(self,x,y,xt=None,yt=None):
        checkpointer = ModelCheckpoint(filepath=modelName, save_best_only=True)
        tensbrd = TensorBoard(log_dir='./tmp/tbLog')
        eStop = EarlyStopping(patience=10)#损失函数不再减小后十轮停止训练
        callback = [checkpointer,eStop,tensbrd]
        
        history = self.Classifer.fit(x,y,epochs=self.n_epochs,batch_size=self.batch_size,validation_split=0.2,callbacks=callback)
        if not xt:
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
    data = input_data.read_data_sets("D:\\4语种DBF50\\")
    nn = FixNN(shape = input_data.img_shape)
    nn.fit(data.train.wavs,data.train.labels)#训练时自动保存
    #nn.load()
    #nn.TestAcc(data.test.wavs,data.test.labels)
    nn.predict(data.test.wavs[:10],data.test.labels[:10])
    a=input('训练结束')