# Developing a face recognition system

import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct
# import face_recognition as fr
import numpy as np
import pickle
import random
import cv2


# -*-The First algorithm -*-
# Create creat
def create_data():
    knownInformation = []
    knownNames = []
    i = 1
    while i <= 40:
        j = 1
        while j <= im_count:
            im = cv2.imread('archive\\s%d\\%d.pgm' % (i, j), 0)
            name = "s%d" % i
            sc, ff, dc = scale(im, width, height)
            form = [sc, cut_image(im), random_pxl(im), ff, dc]
            knownInformation.append(form)
            knownNames.append(name)
            j += 1
        i += 1

    inf = {"information": knownInformation, "names": knownNames}
    f = open("data_base", "wb")
    f.write(pickle.dumps(inf))
    f.close()


# Scale, FFT, DCT
def scale(img, w, h):
    inf = []
    inf1 = []
    inf2 = []
    new_h = int(h * scale_percent / 100)
    new_w = int(w * scale_percent / 100)
    output = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # FFT, DCT
    output1 = fft(np.array(output)).real
    output2 = dct(np.array(output), 1)
    for x in range(new_h):
        for y in range(new_w):
            inf.append(output[x][y])
            inf1.append(output1[x][y])
            inf2.append(output2[x][y])
    return np.array(inf), np.array(inf1), np.array(inf2)


# Random
def random_pxl(img):
    inf = []
    for i in range(len(pxl)):
        inf.append(img[pxl[i][0]][pxl[i][1]])
    return np.array(inf)


# Histogram
def histogram(img):
    inf = []
    hist = cv2.calcHist([img], [0], None, [his_index], [0, 256])
    for i in range(len(hist)):
        inf.append(hist[i][0])
    return np.array(inf)


# Cut image
def cut_image(image):
    psz_height = height//psz
    psz_width = width // psz
    box_list = []
    for x in range(0, psz):
        for y in range(0, psz):
            cropped = image[x*psz_width:(x+1)*psz_height, y*psz_height:(y+1)*psz_width]
            box_list.append(histogram(cropped))
    return np.array(box_list)


# Multi_function
def multi_fun(et_faced, formed, depd, named, a):
    razed = np.sum(np.absolute(np.subtract(formed, et_faced, dtype=float)))
    if razed < depd:
        return razed, data["names"][a]
    else:
        return depd, named


# Search
def search(arr_names):
    some_name = "NO"
    largest = 0
    for b in range(len(arr_names)):
        idx = 1
        for c in range(len(arr_names)):
            if arr_names[b] == arr_names[c] and b != c:
                match b:
                    case 0:
                        idx += 75
                    case 1:
                        idx += 75
                    case 2:
                        idx += 72
                    case 3:
                        idx += 68
                    case 4:
                        idx += 68
        if largest < idx:
            largest = idx
            some_name = arr_names[b]
    return some_name


def print_img(img, et_img):
    output = cv2.resize(img, (int(width*scale_percent/100), int(height*scale_percent/100)), interpolation=cv2.INTER_LINEAR)
    et_output = cv2.resize(et_img, (int(width * scale_percent / 100), int(height * scale_percent / 100)), interpolation=cv2.INTER_LINEAR)
    vis = np.concatenate((output, et_output), axis=1)
    cv2.imshow('scale', cv2.resize(vis, (width*2, height), interpolation=cv2.INTER_LINEAR))

    plt.hist(cv2.calcHist([img], [0], None, [his_index], [0, 256]))
    plt.hist(cv2.calcHist([et_img], [0], None, [his_index], [0, 256]))
    plt.show()


# Main
def main():
    global data
    not_count = 0
    count = 0
    i = 1
    while i <= 40:
        j = 1
        while j <= 10:
            # Who will be searching
            imgTest = cv2.imread('archive\\s%d\\%d.pgm' % (i, j), 0)
            name = "s%d" % i

            sc, ff, dc = scale(imgTest, width, height)
            form = [sc, cut_image(imgTest), random_pxl(imgTest), ff, dc]

            # Find the best et_name for each method
            arr_names = []
            for t in range(len(form)):
                dep = np.inf
                et_name = "NO"
                a = 0
                while a < im_count*40:
                    et_face = data["information"][a]
                    dep, et_name = multi_fun(et_face[t], form[t], dep, et_name, a)
                    a += 1
                arr_names.append(et_name)
            et_name = search(arr_names)

            # Outputting
            et_img = cv2.imread('archive\\%s\\1.pgm' % et_name, 0)
            print_img(imgTest, et_img)
            twoImage = np.hstack((imgTest, et_img))
            cv2.imshow('searching', twoImage)
            cv2.waitKey(0)

            if et_name == name:
                cv2.imshow('Find', twoImage)
                cv2.waitKey(50)
                count += 1
            else:
                cv2.imshow('not Find', twoImage)
                cv2.waitKey(50)
                not_count += 1
            j += 1
        i += 1

    count = count - im_count*40
    print('Всего найдено точных совпадений %d == %f' % (count, 100/(400-im_count*40)*count) + '%')
    print('Не найдено или совершена ошибка %d == %f' % (not_count, 100/(400-im_count*40)*not_count) + '%')


# The sizes of image
size = cv2.imread('archive\\s1\\1.pgm', 0)
(height, width) = size.shape

# The count of random points
p = 500
# The array of p random point
pxl = [[random.randrange(0, height), random.randrange(0, width)] for i in range(p)]

# The percent of image (20% it is 20/100 = 1/5 of image)
scale_percent = 20
# Histogram index
his_index = 32
# The count of division (4x4 = 16 psz)
psz = 4

# Count of image of each person
im_count = 3
create_data()

# Programme start
data = pickle.loads(open("data_base", "rb").read())
main()

'''# -*-The Second algorithm -*-
knownEncodings = []
knownNames = []
count = 0
not_count = 0

m = 1
while m <= 40:
    name = "s%d" % m
    # загружаем изображение и конвертируем его
    img1 = fr.load_image_file('archive\\s%d\\1.pgm' % m)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # вычисляем для каждого лица
    encodings = fr.face_encodings(img1)[0]

    # составляем массив из данных
    knownEncodings.append(encodings)
    knownNames.append(name)
    m += 1

# создаём базу из первых лиц
data = {"encodings": knownEncodings, "names": knownNames}
f = open("data_base_1", "wb")
f.write(pickle.dumps(data))
f.close()

# Загружаем базу известных лиц
data = pickle.loads(open("data_base_1", "rb").read())
i = 1
while i <= 40:
    j = 2
    while j <= 10:
        # Кого будем искать
        name = "s%d" % i
        imgTest = fr.load_image_file('archive\\s%d\\%d.pgm' % (i, j))
        imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

        try:
            # вычисляем для каждого лица
            enTest = fr.face_encodings(imgTest)[0]

            a = 1
            while a <= 40:
                encodings = data["encodings"][a]
                match = fr.face_distance([encodings], enTest)
                if match and name == data["names"][a]:
                    count += 1
                a += 1

        except:
            not_count += 1

        j += 1
    i += 1

print('Всего найдено точных совпадений %d' % count)
print('Всего комбинаций %d' % not_count)

cv2.waitKey(0)'''
