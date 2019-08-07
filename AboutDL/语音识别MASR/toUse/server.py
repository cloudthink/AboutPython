from flask import Flask, request
import _init_path
from models.conv import GatedConv
import sys
import numpy as np
#import beamdecode

app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    datas = request.json
    token = datas['token']
    receipt_data = list(datas['data'])
    if token == 'SR':
        model = GatedConv.load("语音识别MASR/pretrained/gated-conv.pth")
        text = model.predict(receipt_data)
        return text
    elif token == 'FN':
        nums = np.array(receipt_data)
        mean = np.mean(nums)
        median = np.median(nums)
        return '平均数：{}   中位数：{}'.format(mean,median)


app.run("172.16.100.29", debug=True)
