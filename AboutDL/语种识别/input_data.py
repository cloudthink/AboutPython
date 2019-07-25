import glob
import numpy as np
import random
import librosa
import timeit
from sklearn.model_selection import train_test_split
import os
import pylab

frame_length =50
mfcc_length =30

lab_dict={'isiXhosa':0,'Afrikaans':1,'Sesotho':2,'Setswana':3}#(手动维护)标签数字化字典

rev_lab_dict = {value:key for key,value in lab_dict.items()}#(自动生成)反义字典，用于查看真实标签
rev_ten = np.array([x for x in range(len(lab_dict))])#(自动生成)转义向量：用于将独热化的标签转换为字典数字
catchDirPath = '/media/yangjinming/DATA/Dataset/Catch'

class DataSet(object):
    def __init__(self, x=None,y=None):
        self._wavs = self._labels = None
        self._epochs_completed = self._index_in_epoch = self._num_examples = 0
        if x is not None:
            self._wavs = x
            self._labels = y
            self._num_examples = len(self._wavs)

    def read(self, wavs=None,fileNum = None):
        x , y = [],[]
        if not os.path.exists(catchDirPath):#因为加载文件过慢，所以只在第一次进行原始文件读取，之后都读取缓存文件
            os.makedirs(catchDirPath)
        if fileNum is None:#指定了哪个缓存就去直接加载，在这个数非None的情况下可以不需要原始音频文件
            fileNum = len(wavs)
        #因为多次实验的目的，即便文件数目相同的情况下可以设置不同的MFCC和取多少帧长为一个样本，所以针对缓存文件要更有标识性
        if not os.path.exists(os.path.join(catchDirPath, "{}xZ{}M{}.npy".format(fileNum,frame_length,mfcc_length))):
            for i,wav in enumerate(wavs):
                wave,sr = librosa.load(wav,sr=None,mono = True)
                label = np.eye(len(lab_dict))[lab_dict[wav.split('/')[-1].split('_')[0]]]#默认按独热表示了，需要非独热的化去掉np.eye即可
                mfccTot = librosa.feature.mfcc(wave,sr,n_mfcc=mfcc_length).transpose()#转置，每一行一帧
                for i in range(len(mfccTot)):
                    if i % frame_length==0 and i>0:
                        if i == frame_length:
                            mfcc = mfccTot[0:frame_length].flatten()#降维
                        else:
                            mfcc = mfccTot[i-frame_length:i].flatten()#降维
                    else:
                        continue
                    x.append(np.array(mfcc))
                    y.append(label)
            #均值方差标准化
            x[:] -= np.mean(x,axis=0)
            x[:] /= np.var(x)

            np.save(os.path.join(catchDirPath, "{}xZ{}M{}.npy".format(fileNum,frame_length,mfcc_length)),x)
            np.save(os.path.join(catchDirPath, "{}yZ{}M{}.npy".format(fileNum,frame_length,mfcc_length)),y)
            #for i in range(len(x)//10000+1):#分片保存，当缓存文件过大时可以考虑采用分片，目前看1G只能直接快速加载了不需要分片（分片的话性能提升与否未测试）
            #    if i == len(x):
            #        tempX=x[i:]
            #        tempY=y[i:]
            #    else:
            #        tempX=x[i*10000:(i+1)*10000]
            #        tempY=y[i*10000:(i+1)*10000]
            #    np.save(os.path.join(catchDirPath, "{}x{}.npy".format(fileNum,i)),tempX)
            #    np.save(os.path.join(catchDirPath, "{}y{}.npy".format(fileNum,i)),tempY)
        else:
            x = np.load(os.path.join(catchDirPath, "{}xZ{}M{}.npy".format(fileNum,frame_length,mfcc_length)))
            y = np.load(os.path.join(catchDirPath, "{}yZ{}M{}.npy".format(fileNum,frame_length,mfcc_length)))
            #for i in range(int(fileNum)):#和分片保存对应的分片加载
            #  if os.path.exists(os.path.join(catchDirPath, "{}x{}.npy".format(fileNum,i))):
            #    x.extend(np.load(os.path.join(catchDirPath, "{}x{}.npy".format(fileNum,i))))
            #    y.extend(np.load(os.path.join(catchDirPath, "{}y{}.npy".format(fileNum,i))))
        self._wavs = x
        self._labels = y
        self._num_examples = len(self._wavs)

    @property
    def wavs(self):
        return self._wavs
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed

    #取出下一批数据
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._wavs[start:end], self._labels[start:end]

#外部调用，获取数据集
def read_data_sets(train_dir, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    files = glob.glob(train_dir + "*.wav")#加载出文件夹中全部音频文件
    random.shuffle(files)#对加载出来的文件进行一次打乱，不然每个语种的聚集在一起

    start_time = timeit.default_timer()
    print("#文件数量：{}\n开始读取音频文件特征...".format(len(files)))
    ds = DataSet()
    #ds.read(wavs=files)#读文件用文件列表
    ds.read(fileNum='9821')#当明确知道已经有缓存文件存在时可以直接用对应数目去加载缓存文件，可以删掉脱离原文件
    index = random.sample(range(len(ds.wavs)),len(ds.wavs))#全样本下标打乱，因为音频分帧的话同一个音频拆出来的相同语种的还是挨着的
    print('总样本数量：{}'.format(len(index)))
    V_SIZE,CV_SIZE = int(len(index)*0.9),int(len(index)*0.7)#验证集取数据集的0.1,交叉验证集取数据集的0.2
    indexT,indexCV,indexV = index[:CV_SIZE],index[CV_SIZE:V_SIZE],index[V_SIZE:]
    print("训练集样本数量：{}\n交叉验证集样本数量：{}\n验证集样本数量：{}".format(len(indexT),len(indexCV),len(indexV)))
    data_sets.train = DataSet(ds.wavs[indexT],ds.labels[indexT])
    data_sets.validation = DataSet(ds.wavs[indexCV],ds.labels[indexCV])
    data_sets.test = DataSet(ds.wavs[indexV],ds.labels[indexV])
    print("\n数据处理耗时 {0} 分钟".format((timeit.default_timer() - start_time) / 60))

    return data_sets
