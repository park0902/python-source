from PIL import Image
import os
import re
import matplotlib.image as mimage
import math
import numpy as np
import csv
import cx_Oracle
import time



class OraDB:
    INFO = 'scott/tiger@192.168.19.1:1521/orcl'
    ADMIN = None

    @classmethod
    def createConn(cls,info):
        OraDB.ADMIN = cx_Oracle.connect(info)
        print('DB 커넥션 객체가 생성되었습니다.')

    @classmethod
    def prepareCursor(cls):
        return OraDB.ADMIN.cursor()

    @classmethod
    def dbCommit(cls):
        return OraDB.ADMIN.commit()

    @classmethod
    def releaseConn(cls):
        OraDB.prepareCursor().close()




class ImageToCSV:
    __IMAGE_PATH = 'D:\\data\\python\\x-ray\\image\\'
    __RESIZED_IMAGE_PATH = __IMAGE_PATH + 'resized\\'
    __LABEL_PATH = 'D:\\data\\python\\x-ray\\xray_labels.csv'

    insertCNT = 0
    query = None

    def __init__(self):
        self.__image_data = []  # 이미지가 Gray Scale 로 변환된 데이터.
        self.__number = 1  # 이미지 번호.
        self.__rgb_cnt = 0
        self.__train_labels = self._setLabel()
        self.__labels = {
            'patient': 1,
            'normal': 2
        }
        self.__label_list = {}
        self.seq = 0

    def _setLabel(self):
        file = open('D:\\data\\python\\x-ray\\xray_labels.csv', 'r')
        reader = csv.reader(file, delimiter=',')
        list = ['dummy']   #0번째를 더미변수로 막아서 1부터 append 될 수 있도록 함
        for r in reader:
            list.append(r[0])
        return list

    def _image_to_thumbnail(self):
        '''
            기존 원본 이미지를 특정 사이즈 형식으로 Thumbnail 을 수행하는 함수.
            이미지가 저장된 폴더로부터 이미지를 로드 후 썸네일 이미지 생성.
        '''
        size = (200, 200)   # 32,32 하면 32x32형태의 썸네일 이미지를 만들어준다.
                            # 만일, 위아래 사이즈가 맞지 않다면. 그 부분을 흰색으로 채워준다.

        for file in [filename for filename in os.listdir(ImageToCSV.__IMAGE_PATH) if
                     re.search('[0-9]+\.(jpg|jpeg|png)', filename) is not None]:
            try:
                # print(file)
                filename, ext = os.path.splitext(file)

                new_img = Image.new("RGB", (200, 200), "white")
                im = Image.open(ImageToCSV.__IMAGE_PATH + str(file))  #해당 이미지를 변수에 넣어서 인스턴스화
                im.thumbnail(size, Image.ANTIALIAS)  #지정한 사이즈 (200,200)으로 이미지를 바꿔준다.
                load_img = im.load()
                load_newimg = new_img.load()
                i_offset = (200 - im.size[0]) / 2
                j_offset = (200 - im.size[1]) / 2

                for i in range(0, im.size[0]):
                    for j in range(0, im.size[1]):
                        load_newimg[i + i_offset, j + j_offset] = load_img[i, j]

                if ext.lower() in ('.jpeg', '.jpg'):
                    new_img.save(ImageToCSV.__IMAGE_PATH + 'resized\\' + str(filename) + '.jpeg')
                elif ext.lower() == '.png':
                    new_img.save(ImageToCSV.__IMAGE_PATH + 'resized\\' + str(filename) + '.png')
            except Exception as e:
                print(str(file), e)

    def _rgb2gray(self, rgb):
        '''
            YCrCb : 디지털(CRT, LCDl, PDP 등)을 위해서 따로 만들어둔 표현방법.
             - Y : Red*0.2126 + Green*0.7152 + Blue*0.0722
            YPbPr : 아날로그 시스템을 위한 표현방법.
             - Y : Red*0.299 + Green*0.587 + Blue*0.114
            실제 RGB 값들을 Gray Scale 로 변환하는 함수 .
        '''

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 255 * (0.2126 * r + 0.7152 * g + 0.0722 * b)

        return np.array(gray).astype('int32')  #소수점 버리기 위해 int32로 바꿔줌.



    # png rgb값은 0~1 사이의 소수, jpeg rgb값은 0~255 사이의 정수이므로 255를 곱해주어야 한다!
    def _rgb2gray_png(self, rgb):
        gray = 255 * np.dot(rgb[...,:3], [0.2126, 0.7152, 0.0722])

        return np.array(gray).astype('int32')

    def _extract_rgb_from_image(self):
        '''
            이미지 파일들을 읽어들여 Gray Scale 로 변환하는 함수.
        '''
        labels = enumerate(open(self.__LABEL_PATH, 'r'))
        for key, val in labels:
            self.__label_list[key + 1] = val.replace('\n','')

        # todo 그레이스케일된 이미지 정보 + 라벨 들어있는 곳 : __image_data
        for name in [filename for filename in os.listdir(ImageToCSV.__RESIZED_IMAGE_PATH)]:
            try:
                img = mimage.imread(ImageToCSV.__RESIZED_IMAGE_PATH + str(name))  #img : (200,200,3)
                gray = self._rgb2gray(img)  # rgb상태의 img를 삽입
                gray = gray.flatten()  # 1차원 형태로 변환
                gray = ','.join([str(i) for i in gray.tolist()])   # clob에 넣을 수 있도록 str형태로 바꿔줌 ex) '123,234,0,0,0'
                # 라벨 붙이기
                # f = re.split('[_.]', name) # 0_cat.png -> ['0','cat','png']
                f = re.split('[.]', name)   # 1.png -> ['1', 'png']
                label = self.__label_list[int(f[0])]
                label_num = self.__labels[label]
                self.seq += 1

                self.__image_data.append([self.seq, gray, label_num])  # 1차원 gray데이터와, 라벨을 리스트에 담는다.  #join메소드 사용해서 구분자 넣기.
            except OSError as e:
                print(str(name) + ', 이미지를 식별할 수 없습니다.', e)
                continue


            self.__rgb_cnt += 1
            if self.__rgb_cnt % 10 == 0:   #배치단위가 rgb_cnt이다. 100개단위로 insert를 할 것이다.
                print('배치 ',self.__rgb_cnt/10,'번째')
                self.insertFileToDB()
                self.__image_data.clear()   #변수를 청소해준다.


        #540개라면, 마저 넣지 못한 DB를 넣는 부분.
        self.insertFileToDB()
        self.__image_data.clear()



    def insertFileToDB(self):

        cur = OraDB.prepareCursor()
        now = time.localtime()

        if ImageToCSV.insertCNT == 0:
            ImageToCSV.insertCNT += 1
            ImageToCSV.query = """
            CREATE TABLE py_test (
            idx number(10) PRIMARY KEY, column1 CLOB , label varchar2(50) )
            """
            cur.execute(ImageToCSV.query)  # 생성


        print('== '+'테스트'+' : 데이터를 DB 로 이관 중 ('+str(now.tm_year)+'/'+str(now.tm_mon)+'/'+str(now.tm_mday)+' '+
              str(now.tm_hour)+':'+str(now.tm_min)+':'+str(now.tm_sec)+')')

        try:
            #######################################################################################
            ## Oracle Batch I/O
            #######################################################################################
            # cur.prepare('insert into crawledfiles values(:1, :2, :3)')
            # cur.executemany(None, content_list)


            cur.executemany('insert into py_test values(:IDX, :COLUMN1, :LABEL)', self.__image_data)
            OraDB.dbCommit()
            OraDB.releaseConn()
            print('데이터 삽입 완료 ')

        except Exception as e:
            print(e)

    def play_crawler(self):
        # 이미지를 썸네일로 바꾸어 저장
        print('image to thumbnail start.')
        self._image_to_thumbnail()
        print('image to thumbnail end.')

        # 썸네일로 바꾼 이미지를 gray화해서 csv에 저장
        print('rgb to gray start.')
        self._extract_rgb_from_image()
        print('rgb to gray end.')


crawler = ImageToCSV()
OraDB.createConn(OraDB.INFO)
crawler.play_crawler()