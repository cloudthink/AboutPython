import glob
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import random
import os

#这四个全局变量都需要根据情况调整
lab_dict={'isiXhosa':0,'Afrikaans':1,'Sesotho':2,'Setswana':3}#(手动维护)标签数字化字典
rev_lab_dict = {value:key for key,value in lab_dict.items()}#(自动生成)反义字典，用于查看真实标签
rev_ten = np.array([x for x in range(len(lab_dict))])#(自动生成)转义向量：用于将独热化的标签转换为字典数字


class DataSet(object):
    def __init__(self, fileDic,batch_size=100,shuffle=True):
        self.fileList = glob.glob(fileDic + "*.jpg")#加载出文件夹中全部文件
        #实际样本量的多少等于batch_size*（图片增强数量1+1+5+5+1）
        self.batch_size = batch_size
        self._batch_num = len(self.fileList) // self.batch_size
        self.shuffle = shuffle

    #批数据生成器，返回x和y
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
                    label = np.eye(len(lab_dict))[lab_dict[imgPath.split('/')[-1].split('_')[0]]]#默认按独热表示了，需要非独热的化去掉np.eye即可
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


#外部调用，获取数据集
def read_data_sets(tot_dir):
    class DataSets(object):
        pass
    data_sets = DataSets()
    #这里采用人工将数据集分割的方式，如数据集在DATA文件夹中tot_dir应传入DATA，具体图片文集分为三个文件夹分别仍在里面
    data_sets.train = DataSet(os.path.join(tot_dir,'train'))
    data_sets.validation = DataSet(os.path.join(tot_dir,'CV'))
    data_sets.test = DataSet(os.path.join(tot_dir,'test'))
    return data_sets


def read_data(image):
    img_after_decode = tf.image.decode_image(image)
    dim3_img_after_decode = img_after_decode.eval()
    return dim3_img_after_decode
