import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import wave
import os
import utils
import use


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, path = None):
        #音频波形动态显示，实时显示波形，实时进行离散傅里叶变换分析频域
        self.static = False
        if path is not None and os.path.isfile(path):
            self.static = True
            self.stream = wave.open(path)
            self.rate = self.stream.getparams()[2]
            self.chunk = self.rate / 2
            self.read = self.stream.readframes
        else:
            self.rate = 16000
            self.chunk = 400#25*1000/16000针对语音识别25ms为一块这里相同设置
            p = pyaudio.PyAudio()
            self.stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                            input=True, frames_per_buffer=self.chunk)
            self.read = self.stream.read

        '''
        self.data说明：用来记录一整段话的数据，当听到明显声音开始填充，每次都把整个的内容送给语音识别，以其达到效果为：
        你
        你好
        你好啊
        当一个指定时间内没有明显声音时则清空
        '''
        self.data=[]
        self.resHan=[]#语音识别结果，类型待定

        fig = plt.figure(num='Real-time wave')
        ax1 = fig.add_subplot(2, 1, 1)#两行一列，第一子图
        ax2 = fig.add_subplot(2, 1, 2)#两行一列，第二子图

        self.t = np.linspace(0, self.chunk - 1, self.chunk)
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        self.line1, = ax1.plot([], [], lw=2)
        ax1.set_xlim(0, self.chunk)
        ax1.set_ylim(-10000, 10000)

        ax2.set_xlabel('hz')
        ax2.set_ylabel('y')
        self.line2, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, self.chunk)
        ax2.set_ylim(-50, 100)

        interval = int(1000*self.chunk/self.rate)#更新间隔/ms
        animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)


    def _draw_frame(self, framedata):
        #i = framedata
        x = np.linspace(0, self.chunk - 1, self.chunk)
        if self.static:# 读取静态wav文件波形
            y = np.fromstring(self.read(self.chunk/2 +1), dtype=np.int16)[:-1]
        else:# 实时读取声频
            y = np.fromstring(self.read(self.chunk), dtype=np.int16)
            #默认最短3秒为每段话的间隔 3*1000/25=120：只要说话内容间隔3秒以上即清除之前的
            self.data.append(y)
            if np.array(self.data[-120::1]).flatten().max()<1000:
                if len(self.resHan)>0:#如果有语音识别结果则在最后清理之前输出（这个最后的输出是最完整的一句话）
                    print(self.resHan)
                self.data.clear()
                self.resHan.clear()
            elif len(self.data)%20 == 0:#每0.5秒调用一次
                print('语音识别')
                pass#todo:语音识别
                #self.resHan = **#每次都刷新识别结果，即最后一次的结果会是完整一句话

        # 画波形图(上面的)
        self.line1.set_data(x, y)
        # 汉明窗结果（下面的）
        freqs = np.linspace(0, self.rate / 2, self.chunk / 2 + 1)
        _,_,xfp = utils.get_wav_Feature(wavsignal=y)
        self.line2.set_data(freqs, xfp)
        self._drawn_artists = [self.line1, self.line2]


    def new_frame_seq(self):
        return iter(range(self.t.size))


    def _init_draw(self):
        lines = [self.line1, self.line2]
        for l in lines:
            l.set_data([], [])


if __name__ == "__main__":
    ani = SubplotAnimation()
    plt.show()