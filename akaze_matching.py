# OpenCVとosをインポート
import cv2
import os

total_step_number = 10

for Target_step in range(total_step_number):

    TARGET_FILE = "%s\\0.jpg" % Target_step

    IMG_DIR = "D:\\workspace2021\\programs\\Desktop-MetricLearning-Pytorch\\datasets\\train\\datasets\\"
    IMG_SIZE = (400, 400)

    target_img_path = IMG_DIR + TARGET_FILE
    # ターゲット画像をグレースケールで読み出し
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    # ターゲット画像を200px×200pxに変換
    target_img = cv2.resize(target_img, IMG_SIZE)

    # BFMatcherオブジェクトの生成
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # AKAZEを適用、特徴点を検出
    detector = cv2.AKAZE_create()
    (target_kp, target_des) = detector.detectAndCompute(target_img, None)

    # 出力データを保存するフォルダを作成
    output_dir = "output/akaze_%s/" % Target_step
    os.makedirs(output_dir, exist_ok=True)

    print('TARGET_FILE: %s' % (TARGET_FILE))

    files = os.listdir(IMG_DIR)
    for step_number in range(total_step_number):
        file = IMG_DIR + "%s\\2.jpg" %step_number
        if file == '.DS_Store' or file == TARGET_FILE:
            continue
        # 比較対象の写真の特徴点を検出
        comparing_img_path = file
        try:
            comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
            comparing_img = cv2.resize(comparing_img, IMG_SIZE)
            comparing_kp, comparing_des = detector.detectAndCompute(comparing_img, None)
            # BFMatcherで総当たりマッチングを行う
            matches = bf.match(target_des, comparing_des)
            # 特徴量の距離を出し、平均を取る
            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
        except cv2.error:
            # cv2がエラーを吐いた場合の処理
            ret = "ERROR"

        # print(step_number, ret)
        print(ret)

        # matchesをdescriptorsの似ている順にソートする
        matches = sorted(matches, key=lambda x: x.distance)

        # 検出結果を描画する
        img3 = cv2.drawMatches(target_img, target_kp, comparing_img, comparing_kp, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow("%s_%s.jpg" % (Target_step, step_number), img3)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


        # 検出結果を描画した画像を保存する
        cv2.imwrite("output/akaze_%s/%s_%s.jpg" % (Target_step, Target_step, step_number), img3)
