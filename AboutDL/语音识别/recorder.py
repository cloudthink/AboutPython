import os
import requests
import threading
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import wave

import utils
import use


class FileRecord():
    def __init__(self,CHUNK=400,RATE=16000):
        self.filename = None
        self.allowRecording = False
        self.CHUNK = CHUNK
        self.RATE = RATE
        self.intUI()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.mainloop()


    def intUI(self):
        self.root = tkinter.Tk()
        self.root.title('wav音频录制')
        x = (self.root.winfo_screenwidth()-200)//2
        y = (self.root.winfo_screenheight()-140)//2
        self.root.geometry('200x140+{}+{}'.format(x,y))
        self.root.resizable(False,False)
        self.btStart = tkinter.Button(self.root,text='Start',command=self.start)
        self.btStart.place(x=50,y=20,width=100,height=40)
        self.btStop = tkinter.Button(self.root,text='Stop',command=self.stop)
        self.btStop.place(x=50,y=80,width=100,height=40)


    def start(self):
        self.filename = tkinter.filedialog.asksaveasfilename(filetypes=[('Sound File','*.wav')])
        if not self.filename:
            return
        if not self.filename.endswith('.wav'):
            self.filename = self.filename+'.wav'
        self.allowRecording = True
        self.root.title('正在录音...')
        threading.Thread(target=self.record).start()


    def stop(self):
        self.allowRecording = False
        self.root.title('wav音频录制')

    
    def close(self):
        if self.allowRecording:
            tkinter.messagebox.showerror('正在录音','请先停止录音')
            return
        self.root.destroy()


    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format = pyaudio.paInt16,channels=1,rate = self.RATE,
                        input = True,frames_per_buffer=self.CHUNK)
        wf = wave.open(self.filename,'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.RATE)
        while self.allowRecording:#从录音设备读取数据，直接写入wav文件
            data = stream.read(self.CHUNK)
            wf.writeframes(data)
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.filename = None



class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, path = None,serviceAddress='http://127.0.0.1:20000/'):
        self.yysb = use.SpeechRecognition()
        self.httpService = serviceAddress
        #音频波形动态显示，实时显示波形，实时进行离散傅里叶变换分析频域
        if path is not None and os.path.isfile(path):
            self.stream = wave.open(path)
            self.rate = self.stream.getparams()[2]
            self.chunk = int(self.rate/1000*25)
            self.read = self.stream.readframes
        else:
            self.rate = 16000
            self.chunk = 400#25*16000/1000针对语音识别25ms为一块这里相同设置
            p = pyaudio.PyAudio()
            self.stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                            input=True, frames_per_buffer=self.chunk)
            self.read = self.stream.read

        '''
        self.data说明：
        按时调用时：
        用来记录一整段话的数据，当听到明显声音开始填充，每次都把整个的内容送给语音识别，以期达到效果为：
        你
        你好
        你好啊
        当一个指定时间内没有明显声音时则清空
        自动判断启停时：
        从判断开始的数据开始记录，直到判断停止说话准备清空数据前调用一次API，效果：
        你好啊
        '''
        self.data=np.ndarray(shape=(0), dtype=np.int16)
        self.resHan=[]#语音识别结果，类型待定

        fig = plt.figure(num='Real-time wave')
        ax1 = fig.add_subplot(2, 1, 1)#两行一列，第一子图
        ax2 = fig.add_subplot(2, 1, 2)#两行一列，第二子图

        self.t = np.linspace(0, self.chunk - 1, self.chunk)
        #ax1.set_xlabel('t')
        #ax1.set_ylabel('x')
        self.line1, = ax1.plot([], [], lw=2)
        ax1.set_xlim(0, self.chunk)
        ax1.set_ylim(-10000, 10000)

        self.line2, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, self.chunk)
        ax2.set_ylim(-10, 50)

        interval = int(1000*self.chunk/self.rate)#更新间隔/ms
        animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)


    def _valid(self,check_wav):
        '''
        判断是否开始、停止记录声音的方法，返回布尔结果
        if处可能需要根据情况设计更好的判断条件
        当返回为True时，开始、停止记录声音，False则记录声音
        '''
        if check_wav.max()<30 and check_wav.min()>-30:
            return True
        else:
            return False


    def _draw_frame(self, framedata):
        x = np.linspace(0, self.chunk - 1, self.chunk)
        y = np.fromstring(self.read(self.chunk), dtype=np.int16)
        special_flag = False#特殊判断标记，当最后一段音频不足时赋值为真，主要就是针对读取固定长度音频的情况
        if len(y) == 0:
            return
        if len(y)<self.chunk:
            special_flag = True
        self.data = np.append(self.data,np.array(y))

        #默认最短3秒为每段话的间隔 3*1000/25*400=48000：只要说话内容间隔3秒以上即清除之前的
        if special_flag or self._valid(self.data[-48000::1]):
            print('Start/Stop')
            #修改语音识别调用方式：这种是在开始记录有效声音后直到准备清理数据时最后用完整数据调用一次
            if len(self.data)>self.chunk:#大于chunk是每次都添加了一个chunk的数据，下面才清理，所以满足大于chunk条件则说明是在结束时调用
                if True:#本地方式
                    pin,han = self.yysb.predict(self.data)
                    print('识别拼音：{}'.format(pin))
                else:#发送到服务器的方式
                    han = requests.post(self.httpService, {'token':'bringspring', 'wavs':self.data,'pre_type':'W'})
                    han.encoding='utf-8'
                self.resHan = han#记录用
                print('识别汉字：{}'.format(han))#todo:或者给需要的地方

            self.data.clear()
            self.resHan.clear()
        '''
        elif len(self.data)%16000 == 0 or special_flag:#每1秒调用一次
            if True:#本地方式
                pin,han = self.yysb.predict(self.data)
                print('识别拼音：{}'.format(pin))
            else:#发送到服务器的方式
                han = requests.post(self.httpService, {'token':'bringspring', 'wavs':self.data,'pre_type':'W'})
                han.encoding='utf-8'
            self.resHan = han#每次都刷新识别结果，即最后一次的结果会是完整一句话
            if special_flag:
                print('识别汉字：{}'.format(han))#todo:或者给需要的地方
        '''
        y = np.pad(y,(0,self.chunk-len(y)),'constant')#数据维度需要和坐标维度一致
        # 波形图(上面的)
        self.line1.set_data(x, y)
        # 时频图（下面的）
        freqs = np.linspace(0, self.chunk, self.chunk / 2)
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
    #ani = SubplotAnimation('/media/yangjinming/DATA/Dataset/THCTS30/dev/A11_101.wav')
    #plt.show()
    rec = FileRecord()