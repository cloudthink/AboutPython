#coding=utf-8
import utils
import os

'''
测试用，和test不同，test是自己调用模型然后去测试，use则是测试的整体性，使用语音识别类
'''

if __name__ == "__main__":
    yysb = utils.SpeechRecognition()
    data_args = utils.data_hparams()
    data_args.data_type = 'test'
    test = utils.get_data(data_args)

    for i in range(10):
        yysb.testPinyin(' '.join(test.pny_lst[i]),test.han_lst[i])#拼音的已经可以了
        #yysb.predict(os.path.join(test.data_path,test.wav_lst[i]),test.pny_lst[i],test.han_lst[i],come_from_file=True)
