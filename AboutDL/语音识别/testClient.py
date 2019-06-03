import requests
import os
import utils
import scipy
data_args = utils.data_hparams()
test = utils.get_data(data_args)

_,wav = scipy.io.wavfile.read(os.path.join('/media/yangjinming/DATA/Dataset',test.wav_lst[0]))
datas={'token':'bringspring', 'wavs':wav,'pre_type':'W'}
r = requests.post('http://127.0.0.1:20000/', datas)
r.encoding='utf-8'

print(r.text)