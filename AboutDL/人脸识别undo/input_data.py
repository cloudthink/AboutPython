import glob
import numpy as np
from PIL import Image
import random
import librosa
import timeit
from sklearn.model_selection import train_test_split
import os
import pylab

catchDirPath = 'C:\\DataSet\\Catch'
#这四个全局变量都需要根据情况调整
img_shape = [1024,768,255]
lab_dict={'isiXhosa':0,'Afrikaans':1,'Sesotho':2,'Setswana':3}#(手动维护)标签数字化字典
rev_lab_dict = {value:key for key,value in lab_dict.items()}#(自动生成)反义字典，用于查看真实标签
rev_ten = np.array([x for x in range(len(lab_dict))])#(自动生成)转义向量：用于将独热化的标签转换为字典数字


class DataSet(object):
    def __init__(self, x=None,y=None):
        self._imgs = self._labels = None
        self._epochs_completed = self._index_in_epoch = self._num_examples = 0
        if x is not None:
            self._imgs = x
            self._labels = y
            self._num_examples = len(self._imgs)

    def read(self, imgs = None,fileNum = None):
        x , y = [],[]
        if not os.path.exists(catchDirPath):#因为加载文件过慢，所以只在第一次进行原始文件读取，之后都读取缓存文件
            os.makedirs(catchDirPath)
        if fileNum is None:#指定了哪个缓存就去直接加载，在这个数非None的情况下可以不需要原始文件
            fileNum = len(imgs)
        
        if not os.path.exists(os.path.join(catchDirPath, "{}x.npy".format(fileNum))):
            for i,imgPath in enumerate(imgs):#todo:图像处理
                imageS = []
                imageS.append(Image.open(imgPath))
                imageS.append(imageS[0].rotata(90))#旋转增加样本量
                imageS.append(imageS[0].rotata(180))#旋转增加样本量
                imageS.append(imageS[0].rotata(270))#旋转增加样本量
                label = np.eye(len(lab_dict))[lab_dict[imgPath.split('\\')[-1].split('_')[0]]]#默认按独热表示了，需要非独热的化去掉np.eye即可
                
                for image in imageS:
                    x= 'todo'
                    x.append(np.array(x))
                    y.append(label)
            #数据中心化
            x[:] -= np.mean(x,axis=0)

            np.save(os.path.join(catchDirPath, "{}x.npy".format(fileNum)),x)
            np.save(os.path.join(catchDirPath, "{}y.npy".format(fileNum)),y)
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
            x = np.load(os.path.join(catchDirPath, "{}x.npy".format(fileNum)))
            y = np.load(os.path.join(catchDirPath, "{}y.npy".format(fileNum)))
            #for i in range(int(fileNum)):#和分片保存对应的分片加载
            #  if os.path.exists(os.path.join(catchDirPath, "{}x{}.npy".format(fileNum,i))):
            #    x.extend(np.load(os.path.join(catchDirPath, "{}x{}.npy".format(fileNum,i))))
            #    y.extend(np.load(os.path.join(catchDirPath, "{}y{}.npy".format(fileNum,i))))
        self._imgs = x
        self._labels = y
        self._num_examples = len(self._imgs)

    @property
    def imgs(self):
        return self._imgs
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
        return self._imgs[start:end], self._labels[start:end]

#外部调用，获取数据集
def read_data_sets(train_dir):
    class DataSets(object):
        pass
    data_sets = DataSets()
    files = glob.glob(train_dir + "*.jpg")#加载出文件夹中全部文件
    random.shuffle(files)#对加载出来的文件进行一次打乱

    start_time = timeit.default_timer()
    print("#文件数量：{}\n开始读取文件特征...".format(len(files)))
    ds = DataSet()
    ds.read(imgs=files)#读文件用文件列表
    #ds.read(fileNum='9821')#当明确知道已经有缓存文件存在时可以直接用对应数目去加载缓存文件，可以删掉脱离原文件
    totalNum = len(ds.imgs)
    print('总样本数量：{}'.format(totalNum))

    #index = random.sample(range(totalNum),totalNum)#全样本下标打乱，这是自己分数据集的方法，只分两个用train_test_split了
    #V_SIZE = int(len(index)*0.9)#验证集取数据集的0.1
    #print("训练集样本数量：{}\n交叉验证集样本数量：{}\n验证集样本数量：{}".format(len(indexT),len(indexCV),len(indexV)))
    #data_sets.train = DataSet(ds.imgs[indexT],ds.labels[indexT])
    #data_sets.validation = DataSet(ds.imgs[indexCV],ds.labels[indexCV])
    #data_sets.test = DataSet(ds.imgs[indexV],ds.labels[indexV])

    data_sets.train,data_sets.test = train_test_split(ds.imgs,test_size=0.1,random_state=2019)
    print("\n数据处理耗时 {0} 分钟".format((timeit.default_timer() - start_time) / 60))

    return data_sets
