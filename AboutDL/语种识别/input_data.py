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
lab_dict={'Japanese':0,'Afrikaans':1,'Sesotho':2}#标签数字化字典
rev_lab_dict={0:'Japanese',1:'Afrikaans',2:'Sesotho'}#反义字典，用于查看真实标签
rev_ten = np.array([0,1,2])#转义向量：用于将独热化的标签转换为字典数字
catchDirPath = 'D:\\DataSet\\Catch'

class DataSet(object):
  def __init__(self, x=None,y=None):
    self._wavs = self._labels = None
    self.x = self.y = None
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = 0
    if x is not None:
        self._wavs = x
        self._labels = y
        self._num_examples = len(self._wavs)

  def read(self, wavs):
    x , y , sx , sy = [],[],[],[]
    if not os.path.exists(catchDirPath):#因为加载文件过慢，所以只在第一次进行原始文件读取，之后都读取缓存文件
      os.makedirs(catchDirPath)
    if not os.path.exists(os.path.join(catchDirPath, str(len(wavs))+"x.npy")):
      for i,wav in enumerate(wavs):
        print("正在读取第 {} 个文件".format(i))
        wave,sr = librosa.load(wav,sr=None,mono = True)
        label = np.eye(3)[lab_dict[wav.split('\\')[-1].split('_')[0]]]
        for i in range(len(wave)):
          if i % 2000==0 and i>0:
            if i == 2000:
              source = wave[0:2000]
            else:
              source = wave[i-2000:i]
          else:
            continue
          sx.append(source)
          sy.append(label)

        #pylab.plot(wave)
        #pylab.title(wav.split('\\')[-1].split('_')[0])
        #pylab.grid()
        #pylab.axis([0,len(wave),-1,1])
        #pylab.show()#显示音频原始图谱

        #frames = librosa.util.frame(wave,frame_length=len(wave)//1024,hop_length=1024)#.transpose()转置
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
      x[:] -= np.mean(x,axis=0)
      x[:] /= np.var(x)
      np.save(os.path.join(catchDirPath, str(len(wavs))+"x{}.npy".format('Source')),sx)
      np.save(os.path.join(catchDirPath, str(len(wavs))+"y{}.npy".format('Source')),sy)
      np.save(os.path.join(catchDirPath, str(len(wavs))+"x.npy"),x)
      np.save(os.path.join(catchDirPath, str(len(wavs))+"y.npy"),y)
    else:
      x = np.load(os.path.join(catchDirPath, str(len(wavs))+"x.npy"))
      y = np.load(os.path.join(catchDirPath, str(len(wavs))+"y.npy"))
      sx = np.load(os.path.join(catchDirPath, str(len(wavs))+"x{}.npy".format('Source')))
      sy = np.load(os.path.join(catchDirPath, str(len(wavs))+"y{}.npy".format('Source')))
    self._wavs = x
    self.x = sx
    self.y = sy
    self._labels = y
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = len(self._wavs)
    self._s_num_examples = len(self.x)
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
    random.shuffle(files)

    start_time = timeit.default_timer()
    print("#文件数量：{}\n开始读取音频文件特征...".format(len(files)))
    ds = DataSet()
    ds.read(files)
    index = random.sample(range(len(ds.wavs)),len(ds.wavs))#全样本下标打乱
    print('总样本数量：{}'.format(len(index)))
    V_SIZE = int(len(index)*0.9)#验证集取数据集的0.1
    CV_SIZE = int(len(index)*0.7)#交叉验证集取数据集的0.2
    indexT = index[:CV_SIZE]
    indexCV = index[CV_SIZE:V_SIZE]
    indexV = index[V_SIZE:]
    print("训练集样本数量：{}".format(len(indexT)))
    data_sets.train = DataSet(ds.wavs[indexT],ds.labels[indexT])
    print("交叉验证集样本数量：{}".format(len(indexCV)))
    data_sets.validation = DataSet(ds.wavs[indexCV],ds.labels[indexCV])
    print("验证集样本数量：{}".format(len(indexV)))
    data_sets.test = DataSet(ds.wavs[indexV],ds.labels[indexV])
    end_time = timeit.default_timer()
    print("\n数据处理耗时 {0} 分钟".format((end_time - start_time) / 60))

    return data_sets
