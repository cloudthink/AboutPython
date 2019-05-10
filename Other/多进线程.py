
#多进程：通过队列进行进程通信
from multiprocessing import Process,Queue
from threading import Thread,Lock
from time import sleep
import os

#self._lock = Lock()

def sub_task(string,q):
    #self._lock.acquire()
    while not q.empty():
        print(string+str(q.get(True))+": %s\n"%os.getpid(), end='', flush=True)
        sleep(0.1)
    #self._lock.release()
def main():
    q = Queue()
    for i in range(10):
        q.put(i)
    Process(target=sub_task, args=('Pong:',q, )).start()#Process是多进程
    Thread(target=sub_task, args=('Ping:',q, )).start()#用Thread是多线程
    
if __name__ == '__main__':
    main()
