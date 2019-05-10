#-*- coding: UTF-8 -*-
import socket
import threading
import time

# 创建一个基于IPv4和TCP协议的Socket
# 创建套接字对象并指定使用哪种传输服务
    # family=AF_INET - IPv4地址
    # family=AF_INET6 - IPv6地址
    # type=SOCK_STREAM - TCP套接字
    # type=SOCK_DGRAM - UDP套接字
    # type=SOCK_RAW - 原始套接字
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 监听端口:绑定监听的地址和端口。服务器可能有多块网卡，可以绑定到某一块网卡的IP地址上，
# 也可以用0.0.0.0绑定到所有的网络地址，还可以用127.0.0.1绑定到本机地址。小于1024的端口号必须要有管理员权限才能绑定
s.bind(('127.0.0.1', 9999))

#调用listen()方法开始监听端口，传入的参数指定等待连接的最大数量
s.listen(255)

#每个连接都必须创建新线程（或进程）来处理，否则，单线程在处理连接的过程中，无法接受其他客户端的连接
def tcplink(sock, addr):
    #print('Accept new connection from %s:%s...' % addr)
    while True:
        data = sock.recv(2048)#客户端发来的字符串，每次最多接收2k字节:
        time.sleep(1)#todo模型处理
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send('数据处理结果到时候把模型调用结果扔回去'.encode('utf-8'))
    sock.close()
    #print('Connection from %s:%s closed.' % addr)

#程序通过一个永久循环来接受来自客户端的连接，accept()会等待并返回一个客户端的连接
while True:
    # 接受一个新连接:
    sock, addr = s.accept()
    # 创建新线程来处理TCP连接:
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()
