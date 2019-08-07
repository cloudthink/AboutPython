import _init_path
from models.conv import GatedConv

model = GatedConv.load("语音识别MASR/pretrained/gated-conv.pth")

text = model.predict("/media/yangjinming/DATA/Dataset/PrimeWords/d/d2/d25104a2-6be0-4950-9ec0-42e8e1303492.wav")

print("识别结果:")
print(text)
