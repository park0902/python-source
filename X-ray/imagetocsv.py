from PIL import Image
import os
import re
import matplotlib.image as mimage
import math
import numpy as np
import csv

class ImageToCSV:
    __IMAGE_PATH = 'img\\'
    __RESIZED_IMAGE_PATH = __IMAGE_PATH + 'resized\\'
    __DATA_PATH = 'csv\\'
    __LABEL_PATH = 'cat_labels.csv'

    def __init__(self):
        self.__image_data = []  # 이미지가 Gray Scale 로 변환된 데이터.
        self.__number = 1  # 이미지 번호.
        self.__rgb_cnt = 0
        self.__train_labels = self._setLabel()
        self.__labels = {
            # 'patient': 1,
            # 'normal': 2
            'cat' : 1,
            'dog': 2
        }

    def _setLabel(self):
        file = open(self.__LABEL_PATH, 'r')
        reader = csv.reader(file, delimiter=',')
        list = ['dummy']
        for r in reader:
            list.append(r[0])
        return list

    def _image_to_thumbnail(self):
        '''
            기존 원본 이미지를 특정 사이즈 형식으로 Thumbnail 을 수행하는 함수.
            이미지가 저장된 폴더로부터 이미지를 로드 후 썸네일 이미지 생성.
        '''
        size = (32, 32)
        for file in [filename for filename in os.listdir(self.__IMAGE_PATH) if
                     re.search('[0-9]+\.(jpg|jpeg|png)', filename) is not None]:
            try:
                print(file)
                filename, ext = os.path.splitext(file)

                new_img = Image.new("RGB", (32, 32), "white")
                im = Image.open(self.__IMAGE_PATH + str(file))
                im.thumbnail(size, Image.ANTIALIAS)
                load_img = im.load()
                load_newimg = new_img.load()
                i_offset = (32 - im.size[0]) / 2
                j_offset = (32 - im.size[1]) / 2

                for i in range(0, im.size[0]):
                    for j in range(0, im.size[1]):
                        load_newimg[i + i_offset, j + j_offset] = load_img[i, j]

                if ext.lower() in ('.jpeg', '.jpg'):
                    new_img.save(self.__IMAGE_PATH + 'resized\\' + str(filename) + '.jpeg')
                elif ext.lower() == '.png':
                    new_img.save(self.__IMAGE_PATH + 'resized\\' + str(filename) + '.png')
            except Exception as e:
                print(str(file), e)

    def _rgb2gray(self, rgb, type):
        '''
            YCrCb : 디지털(CRT, LCDl, PDP 등)을 위해서 따로 만들어둔 표현방법.
             - Y : Red*0.2126 + Green*0.7152 + Blue*0.0722
            YPbPr : 아날로그 시스템을 위한 표현방법.
             - Y : Red*0.299 + Green*0.587 + Blue*0.114
            실제 RGB 값들을 Gray Scale 로 변환하는 함수 .
        '''

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        if type == 'png':
            gray = 255 * (0.2126 * r + 0.7152 * g + 0.0722 * b)
        elif type == 'jpg':
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        return np.array(gray).astype('int32')

    def _extract_rgb_from_image(self):
        '''
            이미지 파일들을 읽어들여 Gray Scale 로 변환하는 함수.
        '''
        for name in [filename for filename in os.listdir(self.__RESIZED_IMAGE_PATH)]:
            try:
                img = mimage.imread(self.__RESIZED_IMAGE_PATH + str(name))

                # 라벨 붙이기
                f = re.split('[.]', name)   # 1.png -> ['1', 'png']
                label = self.__train_labels[int(f[0])]
                label_num = self.__labels[label]

                gray = self._rgb2gray(img, f[-1])
                self.__image_data.append([gray, label_num, name])
            except OSError as e:
                print(str(name) + ', 이미지를 식별할 수 없습니다.', e)
                continue

            self.__rgb_cnt += 1
            if self.__rgb_cnt % 1000 == 0:
                self._data_to_file()
                self.__image_data.clear()

        self._data_to_file()
        self.__image_data.clear()
        self.__rgb_cnt = 0

    def _data_to_file(self):
        '''
            Gray Scale 로 변환된 이미지 정보를 파일로 기록하는 함수.
        '''
        print('데이터를 저장하는 중입니다.')
        for data in self.__image_data:
            x_shape, y_shape = data[0].shape
            temp_data = ''
            for x in range(x_shape):
                for y in range(y_shape):
                    if x == 0 and y == 0:
                        temp_data += str(data[0][x][y])
                    else:
                        temp_data += ',' + str(data[0][x][y])
            # label
            temp_data += ',' + str(data[1])

            with open(self.__DATA_PATH + 'image_data_' + str(
                    math.ceil(self.__rgb_cnt / 1000)) + '_' + '.csv', 'a', encoding='utf-8') as f:
                f.write(temp_data + '\n')
        print('데이터 저장이 완료되었습니다.')

    def img_to_csv(self):
        # 이미지를 썸네일로 바꾸어 저장
        print('image to thumbnail start.')
        self._image_to_thumbnail()
        print('image to thumbnail end.')

        # 썸네일로 바꾼 이미지를 gray로 바꾸어 csv로 저장.
        print('rgb to gray start.')
        self._extract_rgb_from_image()
        print('rgb to gray end.')

crawler = ImageToCSV()
crawler.img_to_csv()

