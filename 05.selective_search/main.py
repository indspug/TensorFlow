# -*- coding: utf-8 -*-

# インポート

import selectivesearch
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import sys

# 定数
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
BATCH_SIZE = 50
STEPS = 50
TRAIN_IMAGE_ROOTDIR = './train_data'
TRAIN_IMAGE_DIRS = ['dog', 'cat']  # フォルダ名をラベルとする
TEST_IMAGE_ROOTDIR = './test_data'
TEST_IMAGE_DIRS = ['dog', 'cat']  # フォルダ名をラベルとする
CKPT_DIR = './ckpt'
MODEL_FILENAME = 'model.ckpt'

# モジュール変数
train_images = []  # 学習用データ-画像
train_labels = []  # 学習用データ-ラベル
test_images = []   # テスト用データ-画像
test_labels = []   # テスト用データ-ラベル


#################################################
# Usage出力
#################################################
def print_usage(exe_name):
    print 'Usage: # python %s [image file path]' % exe_name

#################################################
# エントリーポイント
#################################################
if __name__ == '__main__':
  
    argvs = sys.argv
    argc = len(argvs)
  
    if argc != 2:
        print_usage(argvs[0])
        quit()

    # コマンドライン引数でファイルパスを指定
    filepath = argvs[1]

    # 画像読込
    img = skimage.io.imread(filepath)
    img = skimage.transform.resize(img, (64,64), mode='edge')

    print('aaaaaaaaaa')

    # selective search実行
    label, regions = selectivesearch.selective_search(img, scale=64, sigma=0.8, min_size=2)

    print('bbbbbbbbbb')

    # 元画像にselective searchで取得した短形を追加して表示
    candidates = set()
    for reg in regions:
        if reg['rect'] in candidates:
            continue

        if reg['size'] < 100:
            continue

        x, y, w, h = reg['rect']

        #if float(w)*h > float(256)*256*0.95:
            #continue

        #if (float(w)/h > 1.2) or (float(h)/w > 1.2):
            #continue

        print x, y, w, h, reg['size']
        #print w/h, h/w

        candidates.add(reg['rect'])

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()
    #cv2.imshow('TEST', img)


# EOF
