import cv2  #OpenCVのインポート
import matplotlib.pyplot as plt #matplotlib.pyplotのインポート
import os


total_step_number = 10
for step in range(total_step_number):

    for image_count in range(4):
        fname = "D:\\workspace2021\\programs\\Desktop-MetricLearning-Pytorch\datasets\\train\\datasets\\%s\\%s.jpg" % (step, image_count) #1つ目の画像ファイル名

        # 出力データを保存するフォルダを作成
        output_dir = "output/histgram_%s/" % step
        os.makedirs(output_dir, exist_ok=True)

        img_1 = cv2.imread(fname) #画つ目の像を読み出しオブジェクトimg_1に代入

        # RGBそれぞれについてヒストグラムを描画する
        color = ['r','g','b']
        plt.figure()
        for i,col in enumerate(color):
            histr = cv2.calcHist([img_1],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
        plt.savefig(output_dir + "%s_%s.png" %(step, image_count))
        print('output: %s_%s.png' %(step, image_count))
        # plt.show()

        # hist_g_1 = cv2.calcHist([img_1],[2],None,[256],[0,256]) #img_1のR(赤)のヒストグラムを計算
        # plt.plot(hist_g_1,color = "r") #ヒストグラムをプロット
        # plt.savefig("histgram8_0.png")
        # plt.show() #プロットを表示


        # hist_g_2 = cv2.calcHist([img_2],[2],None,[256],[0,256]) #img_2のR(赤)のヒストグラムを計算
        # plt.plot(hist_g_2,color = "r") #ヒストグラムをプロット
        # plt.savefig("histgram8_2.png")
        # plt.show() #プロットを表示


        # comp_hist = cv2.compareHist(hist_g_1, hist_g_2, cv2.HISTCMP_CORREL) #ヒストグラムの比較。比較methodにcv2.HISTCMP_CORRELを使用
        # print(comp_hist) #類似度を表示