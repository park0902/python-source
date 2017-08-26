'''
쓰레드 (Thread)

thread란 하나의 프로세스내에서 진행되는 하나의 실행단위를 뜻하며, 
하나의 프로세스에서 여러 실행단위가 실행되는것을 멀티스레드라고 한다. 

프로세스와 스레드는 모두 프로그램을 수행된다는 공통점을 가지고 있지만, 
프로세스는 윈도우에서 여러 응용프로그램을 각각 실행시키는것처럼 독립적으로 실행되어 독립된 메모리공간을 사용하지만, 
멀티스레드는 하나의 프로세스내에서 여러 스레드들이 프로세스공간의 메모리를 공유하여 사용할수 있다.

파이썬에서는 threading이라는 멀티스레드 기능을 지원하는 모듈을 제공한다.
'''




'''
- 스레드 객체 

스레드를 사용하기 위해서는 일반적으로 threading.Thread를 상속받은 클래스객체를 생성하여 사용하며, 아래 나열된 메서드들을 주로 이용한다. 
생성자를 재정의하는 경우 반드시 Thread.__init__()을 수행해야 한다.

 
    Thread.start()

        스레드를 시작할때 사용함

 
    Thread.run()

        스레드의 주요 동작을 정의함

 
    Thread.join([timeout])

    스레드가 종료되기를 기다린다. 
    timeout이 정의된경우, 최대 정의된 시간(초)만큼 기다린다.
'''



'''
- Lock 객체

    하나의 프로세스에서 2개 이상의 스레드가 동시에 수행되는 경우, 
    스레드간에 프로세스 공간의 메모리를 공유한다. 
    
    만약 하나의 변수에 대해서 2개이상의 스레드가 변경을 가하고자 한다면, 
    어떤 스레드가 먼저 수행되느냐에 따라 결과가 달라질수 있다. 
    이러한 상태를 경쟁상태(race condition)라고 한다. 
    이러한 상황을 방지하기 위해서 파이썬에서는 Lock이라는 동기화 객체를 지원한다.

 
- Lock객체는 locked와 unlocked의 2가지 상태
    
    acquire()와 release()의 두가지 함수만을 제공한다. 


    unlocked상태
        
        acquire()가 호출되면 locked상태로 바뀐다


    locked상태 
    
        release()가 호출되면 unlocked상태로 바뀐다


    locked상태에서 다른스레드가 acquire()를 호출
    
        locked상태의 스레드가 release()할때까지 멈추게 된다.
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





'''
threading의 Thread와 Lock객체를 이용하여 각각 버그를 수정하는 시뮬레이션 프로그램이다.

전역변수로 총 버그개수를 정의해놓고, 
생성된 3개의developer스레드가 각각 0.1초에 하나씩 버그를 해결해 나가게 된다.

버그개수가 0이 되어서 모든 스레드가 종료되면 각자 몇개씩 버그를 해결했는지 출력한다.
'''

# from threading import Thread, Lock
# import time
#
# count = 10
# lock = Lock()
#
# class developer(Thread):
#     def __init__(self, name):
#         Thread.__init__(self)
#         self.name = name
#         self.fixed = 0
#
#     def run(self):
#         global count
#         while 1:
#             lock.acquire()
#             if count > 0:
#                 count -= 1
#                 lock.release()
#                 self.fixed += 1
#                 time.sleep(0.1)
#             else:
#                 lock.release()
#                 break
#
# dev_list = []
#
# for name in ['usr1', 'usr2', 'usr3']:
#     dev = developer(name)
#     dev_list.append(dev)
#     dev.start()
#
# for dev in dev_list:
#     dev.join()
#     print(dev.name, 'fixed', dev.fixed)


# import threading, time
#
#
# def test1():
#     time.sleep(5)
#
#     print('5초지났다.')
#
#
# def test2():
#     for i in range(5):
#         time.sleep(1)
#
#         print("{0}second".format(i))
#
#
# t1 = threading.Thread(target=test1)
#
# t2 = threading.Thread(target=test2)
#
# t1.start()
#
# t2.start()
#






# import threading
#
# class Messenger(threading.Thread):
#     def run(self):
#         for _ in range(10):
#             print(threading.currentThread().getName())
#
# send = Messenger(name='Sending out messages')
# receive = Messenger(name='Receiving messages')
#
# send.start()
# receive.start()













































