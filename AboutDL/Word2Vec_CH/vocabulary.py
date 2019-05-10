import tensorflow as tf
import numpy as np
import collections
import random
import glob
import os

# 出现频率最高的5000字作为字典
vocabulary_size = 5000

filePath = "D:\\汉语语音集\\"
catchDirPath = 'D:\\汉语语音集\\Catch'
catchPath = os.path.join(catchDirPath, "dic.npy")

def read_data(filePath):
    original_data =[]
    if not os.path.exists(catchDirPath):#因为加载文件过慢，机械硬盘大概10分钟，所以只在第一次进行原始文件读取，之后都读取缓存文件
        os.makedirs(catchDirPath)
        files = glob.glob(filePath + "*.txt")#加载出文件夹中全部文本文件(路径)
        for fPath in files:
            with open(fPath,encoding='UTF-8') as file:
                data = [[line for line in lines.strip()] for lines in file]
                for d in data:
                    original_data.extend(d)
        np.save(catchPath,original_data)
    else:
        original_data = np.load(catchPath)
    return original_data


original_words = read_data(filePath)
# len()函数是Python中的内容，用于测试列表中元素的数量
wordsLen = len(original_words)

def build_vocabulary(original_words):
    # 创建一个名为count的列表，
    count = [["unkown", -1]]

    # Counter类构造函数原型__init__(args,kwds)
    # Counter类most_common()函数原型most_common(self,n)
    # extend()函数会在列表末尾一次性追加另一个序列中的多个值(用于扩展原来的列表）
    # 函数原型为extend(self,iterable)
    count.extend(collections.Counter(original_words).most_common(vocabulary_size - 1))

    # dict类构造函数原型__init__(self,iterable,kwargs)
    dictionary = dict()

    # 遍历count，并将count中按频率顺序排列好的单词装入dictionary，word为键
    # len(dictionary)为键值，这样可以在dictionary中按0到4999的编号引用汉字
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()

    # unkown_count用于计数出现频率较低(属于未知)的字
    unkown_count = 0

    # 遍历original_words原始文本列表，该列表并没有将汉字按频率顺序排列好
    for word in original_words:
        if word in dictionary:  # original_words列表中的字是否出现在dictionary中
            index = dictionary[word]  # 取得该单词在dictionary中的编号赋值给index
        else:
            index = 0  # 没有出现在dictionary中的单词，index将被赋值0
            unkown_count += 1  # 计数这些单词

        # 列表的append()方法用于扩充列表的大小并在列表的尾部插入一项
        # 如果用print(data)将data打印出来，会发现这里这里有很多0值
        data.append(index)

    # 将unkown类型的汉字数量赋值到count列表的第[0][1]个元素
    count[0][1] = unkown_count

    # 反转dictionary中的键值和键，并存入另一个字典中
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data_index = 0
data, count, dictionary, reverse_dictionary = build_vocabulary(original_words)
def generate_batch(batch_size, num_of_samples, skip_distance):
    # 汉字序号data_index定义为global变量，global是Python中的命名空间声明
    # 因为之后会多次调用data_index，并在函数内对其进行修改
    global data_index

    # 创建放置产生的batch和labels的容器
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    num_of_sample_words = 2 * skip_distance + 1

    #创建buffer队列，长度为num_of_sample_words，因为generate_batch()函数
    #会被调用多次，所以这里使用buffer队列暂存来自data的编号
    buffer = collections.deque(maxlen=num_of_sample_words)
    for _ in range(num_of_sample_words):
        buffer.append(data[data_index])
        data_index = (data_index + 1)

    # Python中//运算符会对商结果取整
    for i in range(batch_size // num_of_samples):
        #target=1，它在一个三元素列表中位于中间的位置，所以下标为skip_distance值
        #targets_to_avoid是生成样本时需要避免的汉字列表
        target = skip_distance
        targets_to_avoid = [skip_distance]

        tlist = [elem for elem in range(0, num_of_sample_words) if elem not in targets_to_avoid]
        for j,target in enumerate(tlist):
            # i*num_skips+j最终会等于batch_size-1
            # 存入batch和labels的数据来源于buffer,而buffer中的数据来源于data
            # 也就是说，数组batch存储了目标汉字在data中的索引
            # 而列表labels存储了语境汉字(与目标汉字相邻的汉字)在data中的索引
            batch[i * num_of_samples + j] = buffer[skip_distance]
            labels[i * num_of_samples + j, 0] = buffer[target]

        # 在最外层的for循环使用append()函数将一个新的目标汉字入队，清空队列最前面的汉字
        if data_index == wordsLen:
            return None,None
        buffer.append(data[data_index])
        data_index = (data_index + 1)
    return batch, labels
