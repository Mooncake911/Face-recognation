# Developing a face recognition system using PCA

from imutils import paths
import face_recognition as fr
import pickle
import random
import cv2

# -*-The First algorithm -*-
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
    # используем библиотеку Face_recognition для обнаружения лиц
    find_face = fr.face_locations(img1)[0]
    # вычисляем эмбеддинги для каждого лица
    encodings = fr.face_encodings(img1)[0]
    # pисуем рамку
    #cv2.rectangle(img1, (find_face[0], find_face[3]), (find_face[1], find_face[2]), (255, 0, 255), 2)
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
            # используем библиотеку Face_recognition для обнаружения лиц
            find_face = fr.face_locations(imgTest)[0]
            # вычисляем эмбеддинги для каждого лица
            enTest = fr.face_encodings(imgTest)[0]
            #cv2.rectangle(imgTest, (find_face[0], find_face[3]), (find_face[1], find_face[2]), (255, 0, 255), 2)
            # пробиваем по базе
            #results = fr.api.compare_faces(data["encodings"], enTest, tolerance=0.6)
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
print('Не найдено или совершена ошибка %d' % not_count)


# -*-The Second algorithm -*-

'''m = random.randint(1, 40)
n = random.randint(1, 10)
i = random.randint(1, 40)
j = random.randint(1, 10)'''
m = 1
n = 1
i = 1
j = 2

img1 = fr.load_image_file('archive\\s%d\\%d.pgm' % (m, n))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
imgTest = fr.load_image_file('archive\\s%d\\%d.pgm' % (i, j))
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

try:
    fac = fr.face_locations(img1)[0]
    entrain = fr.face_encodings(img1)[0]

    cv2.rectangle(img1, (fac[0], fac[3]), (fac[1], fac[2]), (255, 0, 255), 2)

    loc = fr.face_locations(imgTest)[0]
    enTest = fr.face_encodings(imgTest)[0]
    
    cv2.rectangle(imgTest, (loc[0], loc[3]), (loc[1], loc[2]), (255, 0, 255), 2)

    # comparing
    results = fr.compare_faces([entrain], enTest)
    facDist = fr.face_distance([entrain], enTest)

    print(results)
    print(facDist)
except:
    print("not found")

# display image
cv2.imshow('Picture_from_database s%d.%d' % (m, n), img1)
cv2.imshow('New_test_picture s%d.%d' % (i, j), imgTest)
cv2.waitKey(0)
