# _*_ coding:utf-8 _*_

import os
import sys
import speech
import webbrowser

def callback(phr):
    if phr == '停止交互':
        speech.say("Goodbye. 人机交互即将关闭，谢谢使用")
        speech.stoplistening() 
        sys.exit()
    elif phr == '好累啊':
        speech.say("正在为您打开AV")
        webbrowser.open_new("http://www.youku.com/")
    elif phr == '深度学习':
        speech.say("深度学习是个好东西，语音识别已经有了")
    elif phr == 'cmd':
        speech.say("即将打开CMD")
        os.popen("C:\Windows\System32\cmd.exe")

    # 可以继续用 elif 写对应的自制中文库中的对应操作

while True:
    phr = speech.input()
    print(phr)
    speech.say("You said %s" % phr)
    callback(phr)