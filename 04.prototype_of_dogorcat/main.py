# -*- coding: utf-8 -*-

# インポート
import tensorflow as tf
import numpy
import os
import sys
import cv2
import imagenet_downloader as downloader
import image_editor
import mymodel

# 定数
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
BATCH_SIZE = 50
STEPS = 50
TRAIN_IMAGE_ROOTDIR = './train_data'
TRAIN_IMAGE_DIRS = ['pug', 'persian_cat']  # フォルダ名をラベルとする
TEST_IMAGE_ROOTDIR = './test_data'
TEST_IMAGE_DIRS = ['toy_poodle', 'egyptian_cat']  # フォルダ名をラベルとする
CKPT_DIR = './ckpt'
MODEL_FILENAME = 'model.ckpt'

# モジュール変数
train_images = []  # 学習用データ-画像
train_labels = []  # 学習用データ-ラベル
test_images = []   # テスト用データ-画像
test_labels = []   # テスト用データ-ラベル
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
    
        # TRAIN_IMAGE_DIRS内の全画像を読込み、
        # 画像のリサイズとラベルの作成を行う。
        for filename in files:
      
            # 画像を読み込む。
            filepath = os.path.join(train_image_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
      
            # 画像をモデルの入力サイズにリサイズする。
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.flatten().astype(numpy.float32) / 255.0
            train_images.append(image)
            #numpy.append(train_images, numpy.array(image), axis=0)
      
            # 学習データに対応するラベルを作成する。
            label = numpy.zeros(num_class)
            label[i] = 1.0
            train_labels.append(label)
            #numpy.append(train_labels, image, axis=0)


#################################################
# テスト用データ読込
#################################################
def read_test_image():
    num_class = len(TEST_IMAGE_DIRS)

    # 学習データディレクトリから画像を読込む
    for i, dir in enumerate(TEST_IMAGE_DIRS):
        train_image_dir = os.path.join(TEST_IMAGE_ROOTDIR, dir)
        files = os.listdir(train_image_dir)

        # TEST_IMAGE_DIRS内の全画像を読込み、
        # 画像のリサイズとラベルの作成を行う。
        for filename in files:

            # 画像を読み込む。
            filepath = os.path.join(train_image_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue

            # 画像をモデルの入力サイズにリサイズする。
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.flatten(0).astype(numpy.float32) / 255.0
            test_images.append(image)
            # numpy.append(train_images, numpy.array(image), axis=0)

            # 学習データに対応するラベルを作成する。
            label = numpy.zeros(num_class)
            label[i] = 1.0
            test_labels.append(label)
            # numpy.append(train_labels, image, axis=0)


#################################################
# Usage出力
#################################################
def print_usage(exe_name):
    print 'Usage: # python %s [download or train or test]' % exe_name
    print '  download : downloads train and test data.'
    print '  train    : trains using train data and saves model.'
    print '  test     : tests using test data.'

#################################################
# エントリーポイント
#################################################
if __name__ == '__main__':
  
    argvs = sys.argv
    argc = len(argvs)
  
    if argc != 2:
        print_usage(argvs[0])
        quit()
    
    if argvs[1] == 'download' :

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ダウンロードには時間がかかるので注意
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # pugの画像をダウンロード
        #num_pug_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'pug'), 'n02110958', 2000)
        #print('%d images of pug are downloaded.' % num_pug_image)

        # Miniature poodleの画像をダウンロード
        num_miniature_poodle_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'miniature_poodle'), 'n02113712', 2000)
        print('%d images of miniature poodle are downloaded.' % num_miniature_poodle_image)

        # Large poodleの画像をダウンロード
        num_large_poodle_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'large_poodle'), 'n02113892', 2000)
        print('%d images of large poodle are downloaded.' % num_large_poodle_image)

        # Standard poodleの画像をダウンロード
        num_standar_poodle_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'standard_poodle'), 'n02113799', 2000)
        print('%d images of large poodle are downloaded.' % num_standar_poodle_image)

        # Persian Catの画像をダウンロード
        #num_persian_cat_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'persian_cat'), 'n02123394', 2000)
        #print('%d images of persian cat are downloaded.' % num_persian_cat_image)

        # Tiger catの画像をダウンロード
        num_tiger_cat_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'tiger_cat'), 'n02123159', 2000)
        print('%d images of tiger cat are downloaded.' % num_tiger_cat_image)

        # tomcatの画像をダウンロード
        num_tomcat_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'tomcat'), 'n02122725', 2000)
        print('%d images of tomcat are downloaded.' % num_tomcat_image)

        # toy poodleの画像をダウンロード
        #num_toy_poodle = _image = downloader.download_image(os.path.join(TEST_IMAGE_ROOTDIR, 'toy_poodle'), 'n02113624', 2000)
        #print('%d images of toy poodle are downloaded.' % num_toy_poodle)

        # Egyptian catの画像をダウンロード
        #num_egyptian_cat_image = downloader.download_image(os.path.join(TEST_IMAGE_ROOTDIR, 'egyptian_cat'), 'n02124075', 2000)
        #print('%d images of egyptian cat are downloaded.' % num_egyptian_cat_image )


    elif argvs[1] == 'train':
         
        # 学習用データ読込み
        print('--- Read_train image ---')
        read_train_image()
        print('  %d images read.' % len(train_images))

        # 入力画像をZCA白色化する
        # ZCA白色化したら逆に精度が落ちたのでコメントアウト
        #print('--- zca_whitening ---')
        #train_images = image_editor.zca_whitening(train_images)
        #print(train_images.shape)

        # モデル生成
        print('--- Create model ---')
        num_class = len(TRAIN_IMAGE_DIRS)
        model = mymodel.mymodel(IMAGE_WIDTH, IMAGE_HEIGHT, 3, num_class)
        model.construct()

        # モデル実行
        print('--- Training start ---')
        model.train(train_images, train_labels, STEPS, BATCH_SIZE)
        #model.train(train_images.tolist(), train_labels, STEPS, BATCH_SIZE)

        # モデル保存
        print('--- Save model ---')
        model.save(CKPT_DIR, MODEL_FILENAME)

    elif argvs[1] == 'test':

        # テスト用データ読込み
        print('--- Read_test image ---')
        read_test_image()
        print('  %d images read.' % len(test_images))

        # 入力画像をZCA白色化する
        #print('zca_whitening')
        #test_images = image_editor.zca_whitening(test_images)

        # モデル生成
        print('--- Restore model ---')
        num_class = len(TEST_IMAGE_DIRS)
        model = mymodel.mymodel(IMAGE_WIDTH, IMAGE_HEIGHT, 3, num_class)
        model.load(CKPT_DIR, MODEL_FILENAME)

        # モデル実行
        print('--- Test start ---')
        model.test(test_images, test_labels)
        #model.test(test_images.tolist(), test_labels, BATCH_SIZE)


    else:
        print_usage(argvs[0])
        quit()
        
# EOF
