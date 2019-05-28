#coding=utf-8
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import utils

K_usePB = True
tf_usePB = True

class SpeechRecognition():
    def __init__(self):
        # 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
        data_args = utils.data_hparams()
        self.train_data = utils.get_data(data_args)

        #print('加载声学模型中...')
        if K_usePB:
            self.AM_sess = tf.Session()
            with gfile.FastGFile(os.path.join(utils.cur_path,'logs_am','amModel.pb'), 'rb') as f:#加载模型
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.AM_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='') # 导入计算图 # 需要有一个初始化的过程
                self.AM_sess.run(tf.global_variables_initializer())
            self.AM_x = self.AM_sess.graph.get_tensor_by_name('the_inputs') #此处的x一定要和之前保存时输入的名称一致！
            self.AM_preds = self.AM_sess.graph.get_tensor_by_name('the_labels')
        else:
            from model_speech.cnn_ctc import Am, am_hparams
            am_args = am_hparams()
            am_args.vocab_size = len(self.train_data.pny_vocab)#这里有个坑，需要和训练时的长度一致，需要强烈关注！
            self.am = Am(am_args)
            self.am.ctc_model.load_weights(os.path.join(utils.cur_path,'logs_am/model.h5'))

        #print('加载语言模型中...')
        if tf_usePB:
            self.sess = tf.Session()
            with gfile.FastGFile(os.path.join(utils.cur_path,'logs_lm','lmModel.pb'), 'rb') as f:#加载模型
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.sess.graph.as_default()
                tf.import_graph_def(graph_def, name='') # 导入计算图 # 需要有一个初始化的过程
                self.sess.run(tf.global_variables_initializer())
            self.x = self.sess.graph.get_tensor_by_name('x') #此处的x一定要和之前保存时输入的名称一致！
            self.preds = self.sess.graph.get_tensor_by_name('preds')
        else:#ckpt
            from model_language.transformer import Lm, lm_hparams
            lm_args = lm_hparams()
            lm_args.input_vocab_size = len(self.train_data.pny_vocab)
            lm_args.label_vocab_size = len(self.train_data.han_vocab)
            lm_args.dropout_rate = 0.
            self.lm = Lm(lm_args)
            self.sess = tf.Session(graph=self.lm.graph)
            with self.lm.graph.as_default():
                saver =tf.train.Saver()
            with self.sess.as_default():
                lmPath = tf.train.latest_checkpoint(os.path.join(utils.cur_path,'logs_lm'))
                saver.restore(self.sess, lmPath)


    def predicts_file(self,files,pinyin=None,hanzi=None):
        res = []
        for i,f in enumerate(files):
            p = pinyin[i] if pinyin is not None else None
            h = hanzi[i] if hanzi is not None else None
            res.append(self.predict_file(f,p,h))
        return res


    def predicts(self,wavs,pinyin=None,hanzi=None):
        res = []
        for i,wav in enumerate(wavs):
            p = pinyin[i] if pinyin is not None else None
            h = hanzi[i] if hanzi is not None else None
            res.append(self.predict(wav,p,h))
        return res


    def predict_file(self,file,pinyin=None,hanzi=None):
        x,_ = utils.get_wav_Feature(file)
        return self.predict(x,pinyin,hanzi)


    def predict(self,x,pinyin=None,hanzi=None):
        if K_usePB:
            result = self.AM_sess.run(self.AM_preds, {self.AM_x: x})
        else:
            result = self.am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = utils.decode_ctc(result, self.train_data.pny_vocab)
        text = ' '.join(text)
        print('识别拼音：', text)
        if pinyin is not None:
            print('原文拼音：', ' '.join(pinyin))
        with self.sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([self.train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            if tf_usePB:
                preds = self.sess.run(self.preds, {self.x: x})
            else:
                preds = self.sess.run(self.lm.preds, {self.lm.x: x})
            got = ''.join(self.train_data.han_vocab[idx] for idx in preds[0])
            print('识别汉字：', got)
            if hanzi is not None:
                print('原文汉字：', hanzi)
            return text,got

    
    def testPinyin(self,pinyin):
        with self.sess.as_default():
            text = pinyin.strip('\n').split(' ')
            x = np.array([self.train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            if tf_usePB:
                preds = self.sess.run(self.preds, {self.x: x})
            else:
                preds = self.sess.run(self.lm.preds, {self.lm.x: x})
            got = ''.join(self.train_data.han_vocab[idx] for idx in preds[0])
            print('识别汉字：', got)
            return got



if __name__ == "__main__":
    yysb = SpeechRecognition()
    data_args = utils.data_hparams()
    test = utils.get_data(data_args)

    yysb.testPinyin(test.pny_lst[10])
    print(test.han_lst[10])
    
    for i in range(10):
        yysb.predict_file(test.wav_lst[i],test.pny_lst[i],test.han_lst[i])
