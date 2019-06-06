"""
语音识别API的HTTP服务器程序
"""
import http.server
import urllib
import keras
import utils
import numpy as np

yysb = utils.SpeechRecognition(test_flag=False)
class TestHTTPHandle(http.server.BaseHTTPRequestHandler):  
	def setup(self):
		self.request.settimeout(10)
		http.server.BaseHTTPRequestHandler.setup(self)
	
	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()
		
	def do_GET(self):  
		buf = 'SpeechRecognition API'  
		self.protocal_version = 'HTTP/1.1'   
		
		self._set_response()
		buf = bytes(buf,encoding="utf-8")
		self.wfile.write(buf) 
		
	def do_POST(self):  
		'''
		处理通过POST方式传递过来并接收的语音数据
		通过语音模型和语言模型计算得到语音识别结果并返回
		'''
		#获取post提交的数据  
		datas = self.rfile.read(int(self.headers['content-length']))  
		#datas = urllib.unquote(datas).decode("utf-8", 'ignore') 
		datas = datas.decode('utf-8')
		datas_split = datas.split('&')
		token = ''
		pre_type = 'W'
		wavs = []
		for line in datas_split:
			[key, value]=line.split('=')
			if('wavs' == key and '' != value):
				wavs.append(value)
			elif('pre_type' == key):
				pre_type = value
			elif('token' == key ):
				token = value
			else:
				print(key, value)
			
		if(token != 'bringspring'):
			buf = '403'
			buf = bytes(buf,encoding="utf-8")
			self.wfile.write(buf)  
			return
		
		try:
			if len(wavs)>0:
				wavs = np.array([int(w) for w in wavs])
				r = self.recognize(wavs, pre_type)
			else:
				r = ''
		except BaseException as ex:
			r=str(ex)

		self._set_response()
		
		r = bytes(r,encoding="utf-8")
		self.wfile.write(r)
		
	def recognize(self, wavs,pre_type):
		if pre_type == 'F':#传入的文件
			_,han = yysb.predict_file(wavs)
		elif pre_type == 'W':#传入的音频编码
			_,han = yysb.predict(wavs)
		return han

import socket

class HTTPServerV6(http.server.HTTPServer):
	address_family = socket.AF_INET6

def start_server(ip, port):  
	if(':' in ip):
		http_server = HTTPServerV6((ip, port), TestHTTPHandle)
	else:
		http_server = http.server.HTTPServer((ip, int(port)), TestHTTPHandle)
	print('语音识别服务器已开启')
	try:
		http_server.serve_forever() #设置一直监听并接收请求  
	except KeyboardInterrupt:
		pass
	http_server.server_close()
	print('HTTP server closed')
	
if __name__ == '__main__':
	start_server('', 20000) # For IPv4 Network Only
#start_server('::', 20000) # For IPv6 Network