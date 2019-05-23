import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

cur_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        data_path='/media/yangjinming/DATA/Dataset',
        thchs30=True,#默认只使用最小的数据集
        aishell=False,#最大数据集
        prime=False,#第二小
        stcmd=False,#第三小（比较大了11万音频）
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
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        for file in read_files:
            print('加载 ', file, ' 数据...')
            sub_path = os.path.join(cur_path,'data')
            sub_file = os.path.join(sub_path,file)
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data,ncols=80):#ncols进度条宽度
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))
        if self.data_length:#限长：只取前几个
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]

        print('生成/加载 拼音字典...')
        self.pny_vocab = self.LoadCatch('pny_vocab',path=sub_path)
        if self.pny_vocab is None:
            self.pny_vocab = self.mk_pny_vocab(self.pny_lst)#同样是拼音字典
            self.SaveCatch('pny_vocab',self.pny_vocab,path=sub_path)
        print('拼音字典大小：{}'.format(len(self.pny_vocab)))
        print('生成/加载 汉字字典...')
        self.han_vocab = self.LoadCatch('han_vocab',path=sub_path)
        if self.han_vocab is None:
            self.han_vocab = self.mk_han_vocab(self.han_lst)#和拼音字典是不等长的
            self.SaveCatch('han_vocab',self.han_vocab,path=sub_path)
        print('汉字字典大小：{}'.format(len(self.han_vocab)))


    def LoadCatch(self,kindName,path='data/'):
        kindName = kindName+'_{}_{}{}{}{}.npy'.format(self.data_type,int(self.thchs30),int(self.aishell),int(self.prime),int(self.stcmd))
        if os.path.exists(os.path.join(path,kindName)):
            return np.load(os.path.join(path,kindName)).tolist()
        else:
            return None


    def SaveCatch(self,kindName,value,path='data/'):
        kindName = kindName+'_{}_{}{}{}{}.npy'.format(self.data_type,int(self.thchs30),int(self.aishell),int(self.prime),int(self.stcmd))
        np.save(os.path.join(path,kindName),value)


    def get_am_batch(self):
        index_list = [i for i in range(len(self.wav_lst))]
        while True:
            if self.shuffle == True:
                shuffle(index_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst,label_data_lst = [],[]
                begin = i * self.batch_size
                end = begin + self.batch_size
                if end >len(self.wav_lst):
                    begin-=len(self.wav_lst)
                    end-=len(self.wav_lst)
                sub_list = index_list[begin:end]
                for index in sub_list:
                    fbank = compute_fbank(os.path.join(self.data_path,self.wav_lst[index]))
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.pny_vocab)
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
            input_batch = np.array([self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array([self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    #第一个放的是填充，处理字典生成时未遇见过的数据(用训练生成字典，用其他的去找有可能会发生)
    def pny2id(self, line, vocab):
        return [vocab.index(pny) if pny in vocab else 0 for pny in line]


    def han2id(self, line, vocab):
        return [vocab.index(han) if han in vocab else 0 for han in line]


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


# 对音频文件提取mfcc特征
def compute_mfcc(file):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


# 获取信号的时频图
def compute_fbank(file,fs=None,wavsignal=None):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
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

# 获取音频文件的特征向量
def get_wav_Feature(wav,label=None):
    wav_data = []
    fbank = compute_fbank(wav)
    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
    pad_fbank[:fbank.shape[0], :] = fbank
    pad_wav_data, input_length = wav_padding(wav_data)
    return pad_wav_data, input_length

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
