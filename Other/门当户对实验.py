import random
import threading
import multiprocessing
import time
import heapq

'''
MIT的门当户对实验，超简陋实现
'''

class Person(object):
    def __init__(self,num):
        self._num = num
        self._state = False
        self._list = set()

    def __lt__(self,other):#实现自定义类的排序
        if self.state:
            return True
        elif other.state:
            return True
        else:
            return self.value < other.value

    @property
    def value(self):
        return self._num

    @property
    def state(self):#当前响应状态，已经配对则不应该再是其他人目标
        return self._state

    @state.setter
    def state(self,flag):
        self._state = flag
    
    def ZQ(self,id):#被追求了记录下追求者ID
        if not self.state:
            self._list.add(id)

    def Answer(self):#在追求者列表中寻找最大的
        target = list(self._list)
        target.sort()
        return target[-1]

class ShiYan(object):
    def __init__(self):
        self.que = multiprocessing.Queue()
        self._lock = threading.Lock()
        self.Mans = self.Womans = []
        self.Mans = [Person(x) for x in range(1,100,2)]#男的奇数50个
        random.shuffle(self.Mans)#打乱列表
        self.Womans = [Person(x) for x in range(0,101,2)]#女的偶数50个
        random.shuffle(self.Womans)

    def FinWoman(self,man):
        while True:
            time.sleep(len(self.Womans)*0.01)#模拟找人的消耗
            #target = sorted(self.Womans)[-1]#sorted需要在自定义的类中实习__lt__才能使用，下面的heapq和这句效果等同，任选
            target = heapq.nlargest(1, self.Womans, key=lambda x: x.value)[0]#返回的是list，需要下标取出
            #print("%d号男嘉宾目标：%d号女嘉宾，展开攻势"%(man.value,target.value))
            target.ZQ(man.value)#开始追求但不会立刻得到回答
            fee = random.randint(1,100-man.value)#自身分值越高越有可能得到更短的答复时间
            time.sleep(0.015*fee)#消耗的游说时间，自身数字越大理论上越少
            if target.Answer() == man.value:
                self._lock.acquire()
                self.que.put([man.value,target.value,man.value+target.value])
                self.Mans.remove(man)
                self.Womans.remove(target)
                man.state = True
                target.state = True
                self._lock.release()
                print("恭喜男嘉宾%d|女嘉宾%d成功组合！组合分值%d\n"%(man.value,target.value,man.value+target.value))
                break#找不到对象就在这一直找,反正到最后男神女神们都走了歪瓜裂枣没得选肯定有终止

    def FinMan(self,woman):#目前只采用男士主动出击的模式，该方法暂时无用
        while True:
            target = sorted(self.Mans)[-1]
            time.sleep(len(self.Mans)*0.01)#模拟找人的消耗
            target.ZQ(woman.value)
            fee = random.randint(1,101-woman.value)
            time.sleep(0.2*fee)#消耗的游说时间，自身数字越大理论上越少
            if target.Answer() == woman.value:
                self._lock.acquire()
                self.que.put([woman.value,target.value,woman.value+target.value])
                self.Mans.remove(target)
                self.Womans.remove(woman)
                woman.state = True
                target.state = True
                self._lock.release()
                print("组合编号%d|%d\n"%(woman.value,target.value))
                break

    def Main(self):
        pl =[]
        for man in self.Mans:
            p=threading.Thread(target=self.FinWoman, args=(man,))
            pl.append(p)
            p.start()
        for p in pl:#等待所有子进程都完事再去输出队列
            p.join()
        #因为在每一次配对结束后就直接输出了结果所以这里不显示也可以，主要目的是用来学习队列的使用~
        print('配对结果')
        while not self.que.empty():
            print(self.que.get())

if __name__ == "__main__":
    con = 'Yes'
    while con =='Yes':
        sy = ShiYan()
        sy.Main()
        con = input('再试一次？Yes/No:')