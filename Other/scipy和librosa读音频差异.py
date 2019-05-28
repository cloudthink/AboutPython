from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa
# 音频读取差异对比
fs, audio = wav.read('/media/yangjinming/DATA/Dataset/THCTS30/dev/A2_33.wav')#scipy读取的数字编码是整数
wave,sr=librosa.load('/media/yangjinming/DATA/Dataset/THCTS30/dev/A2_33.wav',sr=None,mono = True)#librosa读取的编码是在±1之间的归一化数据
#经过对相同文件读取后的数据进行观察发现：实际数据应该是一样的，只不过librosa的进行了归一化，fs==sr

#mfcc特征计算：即便两个方法处理相同数据结果也是不同的，这里只记录结果，不讨论原因
mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
mfccl = librosa.feature.mfcc(wave,sr,n_mfcc=26)

#mfcc_feat = mfcc_feat[::3]
#mfcc_feat = np.transpose(mfcc_feat)
