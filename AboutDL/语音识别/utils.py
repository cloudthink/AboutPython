import os
import difflib
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

cur_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))#获取的是文件所在路径，不受终端所在影响
#cur_path = os.getcwd()#获取的是终端所在路径
catchDirPath='/home/yangjinming/DataSet/Catch/YYSB'
piece_size=10
K_usePB = True
tf_usePB = True


def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        data_path='/media/yangjinming/DATA/Dataset',
        thchs30=True,
        aishell=True,
        prime=True,
        stcmd=True,
        batch_size=2,
        data_length=None,
        shuffle=True)
    return params



class get_data():
    def __init__(self, args):
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.source_init()


    def source_init(self):
        read_files = []
        if self.thchs30:
            read_files.append('thchs_{}.txt'.format(self.data_type))
        if self.aishell:
            read_files.append('aishell_{}.txt'.format(self.data_type))
        if self.prime:
            read_files.append('prime_{}.txt'.format(self.data_type))
        if self.stcmd:
            read_files.append('stcmd_{}.txt'.format(self.data_type))
        self.wav_lst,self.pny_lst,self.han_lst = [],[],[]

        for file in read_files:
            print('加载 ', file, ' 数据...')
            sub_path = os.path.join(cur_path,'data')
            sub_file = os.path.join(sub_path,file)
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data,ncols=90):#ncols进度条宽度
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))
        if self.data_length:#限长：只取前几个
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        tmp_pny = tmp_han = None
        datatype = 'A'
        print('生成/加载 拼音字典...')
        self.pny_vocab = LoadCatch('pny_vocab',datatype,sub_path,self.thchs30,self.aishell,self.prime,self.stcmd)
        if self.pny_vocab is None:
            tmp_pny,tmp_han = make_all_file()
            self.pny_vocab = self.mk_pny_vocab(tmp_pny)#拼音字典
            self.SaveCatch('pny_vocab',self.pny_vocab,datatype,sub_path)
        #print('拼音字典大小：{}'.format(len(self.pny_vocab)))#1297

        print('生成/加载 汉字字典...')
        self.han_vocab = LoadCatch('han_vocab',datatype,sub_path,self.thchs30,self.aishell,self.prime,self.stcmd)
        if self.han_vocab is None:
            self.han_vocab = self.mk_han_vocab(tmp_han)#和拼音字典是不等长的
            self.SaveCatch('han_vocab',self.han_vocab,datatype,sub_path)
        #print('汉字字典大小：{}'.format(len(self.han_vocab)))#6314


    def SaveCatch(self,kindName,value,datatype,path):
        if datatype == 'A':
            kindName = kindName+'_A.npy'
        else:
            kindName = kindName+'_{}_{}{}{}{}.npy'.format(datatype,int(self.thchs30),int(self.aishell),int(self.prime),int(self.stcmd))
        np.save(os.path.join(path,kindName),value)


    #没有什么意思，完全可以删掉
    def read_file2catch(self):
        self._PWD,self._PLD,self._IL,self._LL = [],[],[],[]
        if not os.path.exists(catchDirPath):#因为直接处理音频数据批稍微大一点显存就OOM了，所以改成将音频全都处理好存成缓存的形式
            os.makedirs(catchDirPath)
        fileindex = 0
        #使用的数据集不同缓存也不相同，如果还有其他变化情况也可以在缓存文件名称上反应出来
        subName = '{}BS{}A{}P{}S{}T{}'.format(self.data_type,self.batch_size,int(self.aishell),int(self.prime),int(self.stcmd),int(self.thchs30))
        if not os.path.exists(os.path.join(catchDirPath, "PWD_{}_0.npy".format(subName))):
            print('开始将全文件处理成缓存，请耐心等待...')
            index_list = [i for i in range(len(self.wav_lst))]
            if self.shuffle == True:
                shuffle(index_list)
            showList = [ i for i in range(len(self.wav_lst) // self.batch_size)]
            for i in tqdm(showList,ncols=90):
                wav_data_lst,label_data_lst = [],[]
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = index_list[begin:end]
                for index in sub_list:
                    fbank = compute_fbank(file=os.path.join(self.data_path,self.wav_lst[index]))
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = pny2id(self.pny_lst[index], self.pny_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = wav_padding(wav_data_lst)
                pad_label_data, label_length = label_padding(label_data_lst)
                self._PWD.append(pad_wav_data.flatten())#pad_wav_data的shape:batch_size，x,200,1
                self._PLD.append(pad_label_data.flatten())#pad_label_data的shape：batch_size，x
                self._IL.append(input_length)#shape：batch_size，
                self._LL.append(label_length)#shape：batch_size，
                if(len(self._PWD)>=piece_size):#每N个保存一次，并清理掉list，防止一直占用内存；因为这里面一个对应的是一个批的数据，所以实际上是很有内容的，不宜过大
                    #当然这种做法的一个问题是需要先有一次建立缓存的过程，不然之前的数据都清空了怎么行。可以直接执行utils进行建立缓存过程
                    np.save(os.path.join(catchDirPath, "PWD_{}_{}.npy".format(subName,fileindex)),self._PWD)
                    self._PWD.clear()#就PWD大，用完赶紧释放
                    np.save(os.path.join(catchDirPath, "PLD_{}_{}.npy".format(subName,fileindex)),self._PLD)
                    self._PLD.clear()
                    np.save(os.path.join(catchDirPath, "IL_{}_{}.npy".format(subName,fileindex)),self._IL)
                    self._IL.clear()
                    np.save(os.path.join(catchDirPath, "LL_{}_{}.npy".format(subName,fileindex)),self._LL)
                    self._LL.clear()
                    fileindex+=1

            if len(self._PWD)> 0:#分片保存，当缓存文件过大时可以考虑采用分片
                np.save(os.path.join(catchDirPath, "PWD_{}_{}.npy".format(subName,fileindex)),self._PWD)
                self._PWD.clear()
                np.save(os.path.join(catchDirPath, "PLD_{}_{}.npy".format(subName,fileindex)),self._PLD)
                self._PLD.clear()
                np.save(os.path.join(catchDirPath, "IL_{}_{}.npy".format(subName,fileindex)),self._IL)
                self._IL.clear()
                np.save(os.path.join(catchDirPath, "LL_{}_{}.npy".format(subName,fileindex)),self._LL)
                self._LL.clear()
        else:
            while False:#和分片保存对应的分片加载(不进行直接加载，缓存也很大，在生成器按需依次读取文件进内存)
                if os.path.exists(os.path.join(catchDirPath, "PWD_{}_{}.npy".format(subName,fileindex))):
                    self._PWD.extend(np.load(os.path.join(catchDirPath, "PWD_{}_{}.npy".format(subName,fileindex))))
                    self._PLD.extend(np.load(os.path.join(catchDirPath, "PLD_{}_{}.npy".format(subName,fileindex))))
                    self._IL.extend(np.load(os.path.join(catchDirPath, "IL_{}_{}.npy".format(subName,fileindex))))
                    self._LL.extend(np.load(os.path.join(catchDirPath, "LL_{}_{}.npy".format(subName,fileindex))))
                    fileindex+=1
                else:
                    break
        print('缓存生成成功')


    #如果使用read_file2catch即全文件生成缓存的方式则将该方法前的下划线去掉加在另一个同名方法上，不然不会出错但是相当于白处理缓存了
    def _get_am_batch(self):
        i = fileindex = 0
        readNext = True
        subName = '{}BS{}A{}P{}S{}T{}'.format(self.data_type,self.batch_size,int(self.aishell),int(self.prime),int(self.stcmd),int(self.thchs30))
        while True:
            if readNext and os.path.exists(os.path.join(catchDirPath, "PWD_{}_{}.npy".format(subName,fileindex))):
                self._PWD=np.load(os.path.join(catchDirPath, "PWD_{}_{}.npy".format(subName,fileindex)))
                self._PLD=np.load(os.path.join(catchDirPath, "PLD_{}_{}.npy".format(subName,fileindex)))
                self._IL=np.load(os.path.join(catchDirPath, "IL_{}_{}.npy".format(subName,fileindex)))
                self._LL=np.load(os.path.join(catchDirPath, "LL_{}_{}.npy".format(subName,fileindex)))
                fileindex+=1
            else:
                fileindex=0
                continue
            #当初降维保存如今恢复回来
            tempPWD = np.array(self._PWD[i]).reshape(self.batch_size,len(self._PWD[i])//200//self.batch_size,200,1)
            tempPLD = np.array(self._PLD[i]).reshape(self.batch_size,len(self._PLD[i])//self.batch_size)
            tempOP = np.zeros(tempPWD.shape[0], )
            inputs = {'the_inputs': tempPWD,'the_labels': tempPLD,
                'input_length': self._IL[i],'label_length': self._LL[i],}
            outputs = {'ctc': tempOP}
            if i==piece_size-1:
                i = 0
                fileindex +=1
                readNext = True
            else:
                i+=1
                readNext = False
            yield inputs, outputs


    def get_am_batch(self):
        index_list = [i for i in range(len(self.wav_lst))]
        while True:
            if self.shuffle == True:
                shuffle(index_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst,label_data_lst = [],[]
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = index_list[begin:end]
                for index in sub_list:
                    fbank = compute_fbank(file=os.path.join(self.data_path,self.wav_lst[index]))
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = pny2id(self.pny_lst[index], self.pny_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = wav_padding(wav_data_lst)
                pad_label_data, label_length = label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,'the_labels': pad_label_data,
                        'input_length': input_length,'label_length': label_length,}
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs


    def get_lm_batch(self):
        for k in range(len(self.pny_lst) // self.batch_size):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array([pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array([han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch


    #拼音字典是声学模型和语言学模型共用的（声学的标签，语言学的输入）
    def mk_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data,ncols=80):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab


    #生成语言学汉字字典
    def mk_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data,ncols=80):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab


    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len



class SpeechRecognition():
    def __init__(self,test_flag = True):
        # 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
        data_args = data_hparams()
        self.train_data = get_data(data_args)
        self.test_flag = test_flag
        #print('加载声学模型中...')
        if K_usePB:
            self.AM_sess = tf.Session()
            with gfile.FastGFile(os.path.join(cur_path,'logs_am','amModel.pb'), 'rb') as f:#加载模型
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.AM_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='') #导入计算图
                self.AM_sess.run(tf.global_variables_initializer())#需要有一个初始化的过程
            self.AM_x = self.AM_sess.graph.get_tensor_by_name('the_inputs:0') #此处的x一定要和之前保存时输入的名称一致！
            self.AM_preds = self.AM_sess.graph.get_tensor_by_name('dense_2/truediv:0')
        else:
            from model_speech.cnn_ctc import Am, am_hparams
            am_args = am_hparams()
            am_args.vocab_size = len(self.train_data.pny_vocab)#这里有个坑，需要和训练时的长度一致，需要强烈关注！
            self.am = Am(am_args)
            self.am.ctc_model.load_weights(os.path.join(cur_path,'logs_am','model.h5'))

        #print('加载语言模型中...')
        if tf_usePB:
            self.sess = tf.Session()
            with gfile.FastGFile(os.path.join(cur_path,'logs_lm','lmModel.pb'), 'rb') as f:#加载模型
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.sess.graph.as_default()
                tf.import_graph_def(graph_def, name='') # 导入计算图 # 需要有一个初始化的过程
                self.sess.run(tf.global_variables_initializer())
            self.x = self.sess.graph.get_tensor_by_name('x:0') #此处的x一定要和之前保存时输入的名称一致！
            self.preds = self.sess.graph.get_tensor_by_name('preds:0')
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
                lmPath = tf.train.latest_checkpoint(os.path.join(cur_path,'logs_lm'))
                saver.restore(self.sess, lmPath)


    def predict_file(self,file,pinyin=None,hanzi=None):
        x,_,_ = get_wav_Feature(wav=file)
        return self.predict(x,pinyin,hanzi,True)


    def predict(self,x,pinyin=None,hanzi=None,come_from_file=False):
        if come_from_file == False:#来自文件的就不用再处理了
            x,_,_ = get_wav_Feature(wavsignal=x)#需要将原始音频编码处理一下
            
        if K_usePB:
            result = self.AM_sess.run(self.AM_preds, {self.AM_x: x})
        else:
            result = self.am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, self.train_data.pny_vocab)
        text = ' '.join(text)
        if self.test_flag:
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
            if self.test_flag:
                print('识别汉字：', got)
                if hanzi is not None:
                    print('原文汉字：', hanzi)
            return text,got

    
    def testPinyin(self,pinyin,hanzi=None):
        with self.sess.as_default():
            text = pinyin.strip('\n').split(' ')
            x = np.array([self.train_data.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            if tf_usePB:
                preds = self.sess.run(self.preds, {self.x: x})
            else:
                preds = self.sess.run(self.lm.preds, {self.lm.x: x})
            got = ''.join(self.train_data.han_vocab[idx] for idx in preds[0])
            if self.test_flag:
                print('识别汉字：', got)
                if hanzi is not None:
                    print('原文汉字：', hanzi)
            return got



#第一个放的是填充，处理字典生成时未遇见过的数据(用训练生成字典，用其他的去找有可能会发生)
def pny2id(line, vocab):
    return [vocab.index(pny) if pny in vocab else 0 for pny in line]


def han2id(line, vocab):
    return [vocab.index(han) if han in vocab else 0 for han in line]


# 加载字典缓存------------------------------------
def LoadCatch(kindName,datatype,path,T=True,A=True,P=True,S=True):
    if datatype =='A':
        kindName = kindName+'_A.npy'
    else:
        kindName = kindName+'_{}_{}{}{}{}.npy'.format(datatype,int(T),int(A),int(P),int(S))
    if os.path.exists(os.path.join(path,kindName)):
        return np.load(os.path.join(path,kindName)).tolist()
    else:
        return None


# 生成全数据集的拼音和汉字列表
def make_all_file():
    read_files = []
    for datatype in ['train','dev','test']:
        read_files.append('thchs_{}.txt'.format(datatype))
        read_files.append('aishell_{}.txt'.format(datatype))
        read_files.append('prime_{}.txt'.format(datatype))
        read_files.append('stcmd_{}.txt'.format(datatype))
    pny_lst,han_lst = [],[]
    for file in read_files:
        sub_path = os.path.join(cur_path,'data')
        sub_file = os.path.join(sub_path,file)
        with open(sub_file, 'r', encoding='utf8') as f:
            data = f.readlines()
        for line in data:
            _, pny, han = line.split('\t')
            pny_lst.append(pny.split(' '))
            han_lst.append(han.strip('\n'))
    return pny_lst,han_lst



# 对音频文件提取mfcc特征------------------------------------
def compute_mfcc(file):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


# 获取信号的时频图------------------------------------
def compute_fbank(file=None,fs=16000,wavsignal=None):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    if wavsignal is None:
        fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input


def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng // 8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


def label_padding(label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens


# 获取音频文件的特征向量------------------------------------
def get_wav_Feature(wav=None,wavsignal=None):
    if wavsignal is None:
        fbank = compute_fbank(file = wav)
    else:
        fbank = compute_fbank(wavsignal = wavsignal)
    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
    pad_fbank[:fbank.shape[0], :] = fbank
    pad_wav_data, input_length = wav_padding([pad_fbank])
    return pad_wav_data, input_length,fbank.flatten()


# 实时声音转换成训练样本------------------------------------
#wav：直接是解码后的音频，label：对应的汉语拼音，型如['ni3','hao3']
pny_vocab = LoadCatch('pny_vocab','A',os.path.join(cur_path,'data'))
def real_time2data(wav,label):
    fbank = compute_fbank(wavsignal = wav)
    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
    pad_fbank[:fbank.shape[0], :] = fbank
    label = pny2id(label, pny_vocab)
    pad_wav_data, input_length = wav_padding([pad_fbank])
    pad_label_data, label_length = label_padding([label])
    inputs = {'the_inputs': pad_wav_data,'the_labels': pad_label_data,
            'input_length': input_length,'label_length': label_length,}
    outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
    return inputs, outputs


# 错词率------------------------------------
def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost


# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text



if __name__ == "__main__":
    data_args = data_hparams()
    data_args.data_type = 'dev'
    data_args.batch_size = 10#可以将不一次性训练am和lm，同样显存情况下lm的batch_size可以比am的大许多
    train_data = get_data(data_args)
    train_data.read_file2catch()