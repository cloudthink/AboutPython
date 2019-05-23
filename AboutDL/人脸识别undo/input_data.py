import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

#这四个全局变量都需要根据情况调整
img_shape = [1024,768,255]
lab_dict={'isiXhosa':0,'Afrikaans':1,'Sesotho':2,'Setswana':3}#(手动维护)标签数字化字典
rev_lab_dict = {value:key for key,value in lab_dict.items()}#(自动生成)反义字典，用于查看真实标签
rev_ten = np.array([x for x in range(len(lab_dict))])#(自动生成)转义向量：用于将独热化的标签转换为字典数字


class DataSet(object):
    def __init__(self, fileDic,batch_size=100,shuffle=True):
        self.fileList = glob.glob(fileDic + "*.jpg")#加载出文件夹中全部文件
        self.batch_size = batch_size
        self._batch_num = len(self.fileList) // self.batch_size
        self.shuffle = shuffle

    def batchReader(self):
        index_list = [i for i in range(len(self.fileList))]
        while True:
            if self.shuffle == True:
                random.shuffle(index_list)
            for i in range(self._batch_num):
                x,y = [],[]
                begin = i * self.batch_size
                end = begin + self.batch_size
                if end >len(self.fileList):
                    begin-=len(self.fileList)
                    end-=len(self.fileList)
                sub_list = index_list[begin:end]
                for index in sub_list:
                    imgPath = self.fileList[index]
                    label = np.eye(len(lab_dict))[lab_dict[imgPath.split('\\')[-1].split('_')[0]]]#默认按独热表示了，需要非独热的化去掉np.eye即可
                    img = tf.gfile.FastGFile(imgPath,'r').read()
                    with tf.Session() as sess:
                        img_after_decode = tf.image.decode_image(img)
                        dim3_img_after_decode = img_after_decode.eval()
                        #查看解码之后的三维矩阵
                        #print(dim3_img_after_decode)
                        #plt.imshow(dim3_img_after_decode)
                        #plt.show()
                        x.append(np.array(dim3_img_after_decode))
                        y.append(label)

                        #随机左右翻转
                        flipped = tf.image.random_flip_left_right(img_after_decode)
                        x.append(np.array(flipped.eval()))
                        y.append(label)

                        #随机亮度调整,max_delta不能为负，在[-max_delta,max_delta]之间随机调整图像亮度
                        for _ in range(5):#取5个随机调整的结果
                            brightness = tf.image.random_brightness(img_after_decode,max_delta=1)
                            x.append(np.array(brightness.eval()))
                            y.append(label)

                        #随机对比度调整,在[lower，upper]之间随机调整对比度，两个数都不能为负
                        for _ in range(5):#取5个随机调整的结果
                            contrast = tf.image.random_contrast(img_after_decode,0.2,18,)
                            x.append(np.array(contrast.eval()))
                            y.append(label)

                        #随机色相调整
                        hue = tf.image.random_hue(img_after_decode,max_delta=0.5)
                        x.append(np.array(hue.eval()))
                        y.append(label)
                        yield x, y
                        #img_after_decode = tf.image.convert_image_dtype(img_after_decode,dtype=tf.float32)

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
