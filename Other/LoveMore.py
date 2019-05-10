import random

boy = 0
girl = 0

def Meet():
    i = 0
    global boy
    global girl
    while True:
        boy = random.randint(1,100)
        girl = random.randint(1,100)
        if boy > 60 and girl >60:
            #print("男孩对女孩喜欢程度：%d\t\t女孩对男孩喜欢程度：%d"%(boy,girl))
            break
        else:
            i += 1
            #print("男孩女孩的第%d次相遇，并没有达成恋爱关系\n"%i)

def MakeLove():
    global boy
    global girl
    boy += random.randint(1,20)
    girl += random.randint(1,20)

def Argue():
    a = boy - random.randint(1,10)
    b = girl - random.randint(1,10)
    if  a>b :
        print('该轮争吵：男孩输了\n')
    elif a < b:
        print('该轮争吵：女孩输了\n')

def ShowAnswer():
    print("男孩对女孩喜欢程度：%d\t\t女孩对男孩喜欢程度：%d"%(boy,girl))

if __name__ == "__main__":
    con='Yes'
    while con == 'Yes':
        Meet()
        MakeLove()
        for i in range(10):
            Argue()
        ShowAnswer()
        print("再玩一次？Yes / No")
        con = input('选择：')
