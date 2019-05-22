#coding=utf-8
import os
import difflib
import glob
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance,get_wav_Feature
# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
# 2.语言模型-------------------------------------------
from model_language.transformer import Lm, lm_hparams

class SpeechRecognition():
    def __init__(self):
        # 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
        from utils import get_data, data_hparams
        data_args = data_hparams()
        self.train_data = get_data(data_args)

        bath = self.train_data.get_am_batch()
        a = next(bath)

        am_args = am_hparams()
        am_args.vocab_size = len(self.train_data.am_vocab)
        self.am = Am(am_args)
        #print('加载声学模型中...')
        self.am.ctc_model.load_weights('logs_am/model.h5')
        
        lm_args = lm_hparams()
        lm_args.input_vocab_size = len(self.train_data.pny_vocab)
        lm_args.label_vocab_size = len(self.train_data.han_vocab)
        lm_args.dropout_rate = 0.
        #print('加载语言模型中...')
        self.lm = Lm(lm_args)
        self.sess = tf.Session(graph=self.lm.graph)
        with self.lm.graph.as_default():
            saver =tf.train.Saver()
        with self.sess.as_default():
            latest = tf.train.latest_checkpoint('logs_lm')
            saver.restore(self.sess, latest)


    def predect(self,x,pinyin=None,hanzi=None):
        x,_ = get_wav_Feature(x)#x为wav音频,转为特征向量
        result = self.am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, self.train_data.am_vocab)
        text = ' '.join(text)
        print('识别拼音：', text)
        if pinyin is not None:
            print('原文拼音：', ' '.join(pinyin))
        with self.sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([self.train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            preds = self.sess.run(self.lm.preds, {self.lm.x: x})
            got = ''.join(self.train_data.han_vocab[idx] for idx in preds[0])
            print('识别汉字：', got)
            if hanzi is not None:
                print('原文汉字：', hanzi)
            return text,got

if __name__ == "__main__":
    yysb = SpeechRecognition()
    from utils import get_data, data_hparams
    data_args = data_hparams()
    test = get_data(data_args)
    for i in range(10):
        yysb.predect(test.wav_lst[i],test.pny_lst[i],test.han_lst[i])
