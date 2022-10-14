#OpenCVをインポート
import cv2

IMG_SIZE = (400, 400)

# １枚目の画像を読み込む
img1 = cv2.imread("D:\\workspace2021\\programs\\Desktop-MetricLearning-Pytorch\datasets\\train\\datasets\\3\\2.jpg")
# ターゲット画像を200px×200pxに変換
img1 = cv2.resize(img1, IMG_SIZE)
# ２枚目の画像を読み込む
img2 = cv2.imread("D:\\workspace2021\\programs\\Desktop-MetricLearning-Pytorch\\datasets\\train\\datasets\\3\\3.jpg")
# ターゲット画像を200px×200pxに変換
img2 = cv2.resize(img2, IMG_SIZE)


# １枚目の画像をグレースケールで読み出し
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# ２枚目の画像をグレースケールで読み出し
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# AKAZE検出器の生成
akaze = cv2.AKAZE_create()
# gray1にAKAZEを適用、特徴点を検出
kp1, des1 = akaze.detectAndCompute(gray1,None)
# gray2にAKAZEを適用、特徴点を検出
kp2, des2 = akaze.detectAndCompute(gray2,None)

# BFMatcherオブジェクトの生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptorsを生成
matches = bf.match(des1, des2)

# matchesをdescriptorsの似ている順にソートする
matches = sorted(matches, key = lambda x:x.distance)

# 検出結果を描画する
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("output.jpg", img3)
cv2.waitKey()

#検出結果を描画した画像を保存する
cv2.imwrite("3_3.jpg", img3)