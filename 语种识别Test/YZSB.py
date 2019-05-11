import glob
import numpy as np
import random
import librosa
import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.layers import LSTM,Dense,Dropout,Flatten,Conv2D,Activation,MaxPooling2D
#from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

class FixNN(object):
    def __init__(self,learning_rate = 0.01,batch_size=64,n_epochs = 1000,dropout = 0.5):
        self.lab_dict={'Japanese':0,'Afrikaans':1,'Sesotho':2}
        self.lr = learning_rate
        self.batch_size=batch_size
        self.n_epochs = n_epochs
        self.dropout=dropout

    def fit(self,x,xc,y,yc,xt,yt):
        x= np.concatenate((x,xc))
        y= np.concatenate((y,yc))
        x = x.reshape(len(x),100,1)
        #y = np_utils.to_categorical(y,3)
        #yt = np_utils.to_categorical(yt,3)
        input_shape = x.shape[1:]
        model = Sequential() 
        model.add(LSTM(500,return_sequences=True,input_shape=input_shape,dropout=self.dropout))
        model.add(LSTM(500,return_sequences=True,input_shape=input_shape,dropout=self.dropout))
        model.add(LSTM(500,return_sequences=True,input_shape=input_shape,dropout=self.dropout))
        #model.add(LSTM(1024,return_sequences=True,input_shape=input_shape,dropout=self.dropout))
        model.add(Flatten())
        model.add(Dense(1024,activation = 'relu'))
        model.add(Dense(1024,activation = 'relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(3,activation='softmax'))
        opt = Adam(lr=self.lr)
        model.compile(loss=keras.losses.MAE,optimizer=opt,metrics=['accuracy'])
        model.build()
        checkpointer = ModelCheckpoint(filepath="yysb.h5", save_best_only=True)
        #tensbrd = TensorBoard(log_dir='./tmp/tbLog')
        eStop = EarlyStopping()#损失函数不再减小后十轮停止训练
        callback = [checkpointer,eStop]
        history = model.fit(x,y,validation_split=0.2,epochs=500,batch_size =self.batch_size,verbose=1,callbacks=callback)
        print(history)

        plt.plot(np.arange(len(history.history['acc'])),history.history['acc'],label='训练集')
        plt.plot(np.arange(len(history.history['val_acc'])),history.history['val_acc'],label='验证集')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.legend(loc=0)
        plt.show()

        loss,acc = model.evaluate(xt,yt)
        print("损失值: %f 精确度: %f"%(loss,acc))
        result = model.predict(xt,batch_size=self.batch_size,verbose=2)

    def CNNfit(self,x,xc,y,yc,xt,yt):
        x= np.concatenate((x,xc))
        y= np.concatenate((y,yc))
        x = x.reshape(len(x),30,43,1)
        xt = xt.reshape(len(xt),30,43,1)
        model = Sequential()
        model.add(Conv2D(50, (3, 3), padding='same', input_shape=x.shape[1:]))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(60, (3, 3), padding='same', input_shape=x.shape[1:]))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(70, (3, 3), padding='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(256,activation = 'relu'))
        model.add(Dropout(0.5))

        model.add(Dense(3,activation = 'softmax'))
        #model.summary()

        opt = keras.optimizers.rmsprop(lr=0.05, decay=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        history = model.fit(x,y,validation_split=0.2,epochs=500,batch_size =self.batch_size,verbose=1,shuffle=True)

        plt.plot(np.arange(len(history.history['acc'])),history.history['acc'],label='训练集')
        plt.plot(np.arange(len(history.history['val_acc'])),history.history['val_acc'],label='验证集')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.legend(loc=0)
        plt.show()

        loss,acc = model.evaluate(xt,yt)
        print("损失值: %f 精确度: %f"%(loss,acc))

if __name__ == "__main__":
    data = input_data.read_data_sets("D:\\DataSet\\", one_hot=True)
    nn = FixNN()
    nn.CNNfit(data.train3.wavs,data.validation3.wavs,data.train3.labels,data.validation3.labels,data.test3.wavs,data.test3.labels)
    #nn.fit(data.train3.wavs,data.validation3.wavs,data.train3.labels,data.validation3.labels,data.test3.wavs,data.test3.labels)