import cv2
import os


Target_step = 9

TARGET_FILE = "%s\\1.jpg" %Target_step
IMG_DIR = "D:\\workspace2021\\programs\\Desktop-MetricLearning-Pytorch\\datasets\\train\\datasets\\"
IMG_SIZE = (200, 200)

target_img_path = IMG_DIR + TARGET_FILE
target_img = cv2.imread(target_img_path)
target_img = cv2.resize(target_img, IMG_SIZE)


# \\\\コード改良の必要あり、現状Rのヒストグラムでしか計算できていない!!!!\\\\

# for i,col in enumerate(color):
#     histr = cv2.calcHist([img_1],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)




target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

print('TARGET_FILE: %s' % (TARGET_FILE))

total_step_number = 10

files = os.listdir(IMG_DIR)
for step_number in range(total_step_number):
    file = IMG_DIR + "%s\\3.jpg" %step_number
    if file == '.DS_Store' or file == TARGET_FILE:
        continue

    comparing_img_path = file

    comparing_img = cv2.imread(comparing_img_path)
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

    ret = cv2.compareHist(target_hist, comparing_hist, 0)
    # print(file, ret)
    print(ret)