# -*- coding: utf-8 -*-

# インポート
import tensorflow as tf
import numpy
import os
import sys
import cv2
import imagenet_downloader as downloader
import model_saver
import image_editor
import mymodel

# 定数
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
TRAIN_IMAGE_ROOTDIR = './data'
TRAIN_IMAGE_DIRS = ['pug', 'persian_cat']  # フォルダ名をラベルとする
CKPT_DIR = './ckpt'
MODEL_FILENAME = 'model.ckpt'

# モジュール変数
train_images = []  # 学習データ-画像
train_labels = []  # 学習データ-ラベル
#train_images = numpy.array([])  # 学習データ-画像
#train_labels = numpy.array([])  # 学習データ-ラベル

#################################################
# 学習用データ読込
#################################################
def read_train_image():
  
    num_class = len(TRAIN_IMAGE_DIRS)

    # 学習データディレクトリから画像を読込む
    for i, dir in enumerate(TRAIN_IMAGE_DIRS):
        train_image_dir = os.path.join(TRAIN_IMAGE_ROOTDIR, dir)
        files = os.listdir(train_image_dir)
    
        # TRAIN_DIMAGE_DIRS内の全画像を読込み、
        # 画像のリサイズとラベルの作成を行う。
        for filename in files:
      
            # 画像を読み込む。
            filepath = os.path.join(train_image_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
      
            # 画像をモデルの入力サイズにリサイズする。
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            #image = image.flatten(1).astype(numpy.float32)
            train_images.append(image)
            #numpy.append(train_images, numpy.array(image), axis=0)
      
            # 学習データに対応するラベルを作成する。
            label = numpy.zeros(num_class)
            label[i] = 1
            train_labels.append(label)
            #numpy.append(train_labels, image, axis=0)


#################################################
# エントリーポイント
#################################################
if __name__ == '__main__':
  
    argvs = sys.argv
    argc = len(argvs)
  
    if argc != 2:
        print 'Usage: # python %s [download,train,test]' % argvs[0]
        quit()
    
    if argvs[1] == 'download' :
         # pugの画像をダウンロード
         num_pug_image = downloader.download_image('pug', 'n02110958', 2000)
         print('%d images of pug are downloaded.' % num_pug_image)
  
         # Persian Catの画像をダウンロード
         num_persian_cat_image = downloader.download_image('persian_cat', 'n02123394', 2000)
         print('%d images of persian cat are downloaded.' % num_persian_cat_image)
    
    elif argvs[1] == 'train':
         
        # 学習用データ読込み
        print('read_train_image')
        read_train_image()
  
        # 入力画像をZCA白色化する
        print('zca_whitening')
        train_images = image_editor.zca_whitening(train_images)
        print(train_images.shape)

        # モデル生成
        print('Create model')
        num_class = len(TRAIN_IMAGE_DIRS)
        model = mymodel.mymodel(IMAGE_WIDTH, IMAGE_HEIGHT, 3, num_class)

        # モデル実行
        print('Iteration start')
        model.construct()
        print(type(train_images))
        print(type(train_labels))
        model.iteration(train_images.tolist(), train_labels)

        # モデル保存
        model_saver.save_model(CKPT_DIR, MODEL_FILENAME)
  
    else:
        print 'Usage: # python %s [download,train,test]' % argvs[0]
        quit()
        
# EOF
