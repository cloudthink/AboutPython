import requests
import os
import utils

data_args = utils.data_hparams()
test = utils.get_data(data_args)

wav,_,_=utils.get_wav_Feature(wav=os.path.join('/media/yangjinming/DATA/Dataset',test.wav_lst[0]))
wav = ','.join([str(f) for f in wav.flatten()])

datas={'token':'bringspring', 'wavs':wav,'pre_type':'W'}
r = requests.post('http://127.0.0.1:20000/', datas)
r.encoding='utf-8'

print(r.text)