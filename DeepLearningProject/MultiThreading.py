'''
쓰레드 (Thread)

파이썬 프로그램은 기본적으로 하나의 쓰레드(Single Thread)에서 실행된다. 
즉, 하나의 메인 쓰레드가 파이썬 코드를 순차적으로 실행한다. 

코드를 병렬로 실행하기 위해서는 별도의 쓰레드(Subthread)를 생성해야 하는데, 
파이썬에서 쓰레드를 생성하기 위해서는 threading 모듈 (High 레벨) 혹은 thread 모듈 (Low 레벨)을 사용할 수 있다. 

일반적으로 쓰레드 처리를 위해서는 thread 모듈 위에서 구현된 threading 모듈을 사용하고 있으며, 
thread 모듈은 (deprecate 되어) 거의 사용하고 있지 않다. 

파이썬(오리지날 파이썬 구현인 CPython)은 전역 인터프리터 락킹(Global Interpreter Lock) 때문에 특정 시점에 하나의 파이썬 코드만을 실행하게 되는데, 
이 때문에 파이썬은 실제 다중 CPU 환경에서 동시에 여러 파이썬 코드를 병렬로 실행할 수 없으며 인터리빙(Interleaving) 방식으로 코드를 분할하여 실행한다. 

다중 CPU 에서 병렬 실행을 위해서는 다중 프로세스를 이용하는 multiprocessing 모듈을 사용한다. 
'''

# from threading import Thread
# import time
#
# def do_work(start, end, result):
#     sum = 0
#     for i in range(start,end):
#         sum += i
#     result.append(sum)
#     return
#
# if __name__=='__main__':
#     s_time = time.time()
#     START, END = 0, 80000000
#     result = list()
#     th1 = Thread(target=do_work, args=(START, END, result))
#     th1.start()
#     th1.join()
# print ('Result : ',sum(result),'time =',time.time()-s_time)


## Result :  3199999960000000 time = 10.740150928497314






# from threading import Thread
# import time
#
# def do_work(start, end, result):
#     sum = 0
#     for i in range(start,end):
#         sum += i
#     result.append(sum)
#     return
#
# if __name__=='__main__':
#     s_time = time.time()
#     START, END = 0, 80000000
#     result = list()
#     # th1 = Thread(target=do_work, args=(START, END, result))
#     th1 = Thread(target=do_work, args=(START, 40000000, result))
#     th2 = Thread(target=do_work, args=(40000000, 80000000, result))
#     th1.start()
#     th2.start()
#     th1.join()
#     th2.join()
# print ('Result : ',sum(result),'time =',time.time()-s_time)





# from threading import Thread
# import time
#
# def do_work(start, end, result):
#     sum = 0
#     for i in range(start,end):
#         sum += i
#     result.append(sum)
#     return
#
# if __name__=='__main__':
#     s_time = time.time()
#     START, END = 0, 80000000
#     result = list()
#     th1 = Thread(target=do_work, args=(START, 20000000, result))
#     th2 = Thread(target=do_work, args=(20000000, 40000000, result))
#     th3 = Thread(target=do_work, args=(40000000, 60000000, result))
#     th4 = Thread(target=do_work, args=(60000000, 80000000, result))
#     th1.start()
#     th2.start()
#     th3.start()
#     th4.start()
#     th1.join()
#     th2.join()
#     th3.join()
#     th4.join()
# print ('Result : ',sum(result),'time =',time.time()-s_time)



from threading import Thread, Lock
import time

count = 10
lock = Lock()

class developer(Thread):
    def __init__(self, name):
        Thread.__init__(self)
        self.name = name
        self.fixed = 0

    def run(self):
        global count
        while 1:
            lock.acquire()
            if count > 0:
                count -= 1
                lock.release()
                self.fixed += 1
                time.sleep(0.1)
            else:
                lock.release()
                break

dev_list = []

for name in ['usr1', 'usr2', 'usr3']:
    dev = developer(name)
    dev_list.append(dev)
    dev.start()

for dev in dev_list:
    dev.join()
    print(dev.name, 'fixed', dev.fixed)

















































