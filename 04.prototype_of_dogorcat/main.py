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
import shutil

# 定数
IMAGE_WIDTH = 32   # 入力画像の幅
IMAGE_HEIGHT = 32   # 入力画像の高さ
COLOR_CHANNELS = 3  # カラーチャンネルの数
RATE_INCREASE = 2   # 前処理して画像1枚をN枚に増やす
BATCH_SIZE = 10
STEPS = 1000         # 学習ステップ
SAVE_STEP = 100     # 一定ステップごとにモデルを保存する
TRAIN_IMAGE_ROOTDIR = './train_data'
TRAIN_IMAGE_DIRS = ['dog', 'cat']  # フォルダ名をラベルとする
#TRAIN_IMAGE_DIRS = ['resized_dog', 'resized_cat']  # フォルダ名をラベルとする
#TRAIN_IMAGE_DIRS = ['selected_golden_retriver', 'selected_tigercat']  # フォルダ名をラベルとする
RESIZED_IMAGE_DIRS = ['resized_dog', 'resized_cat']  # フォルダ名をラベルとする
#TRAIN_IMAGE_DIRS = ['test_dog', 'test_cat']  # フォルダ名をラベルとする
TEST_IMAGE_ROOTDIR = './test_data'
TEST_IMAGE_DIRS = ['dog', 'cat']  # フォルダ名をラベルとする
#TEST_IMAGE_DIRS = ['selected_labrador_retriver', 'selected_tomcat']  # フォルダ名をラベルとする
CKPT_DIR = './ckpt'
MODEL_FILENAME = 'model.ckpt'
#MODEL_FILENAME = 'model.ckpt-1000-500'

# モジュール変数
#train_images = []  # 学習用データ-画像
#train_labels = []  # 学習用データ-ラベル
#test_images = []   # テスト用データ-画像
#test_labels = []   # テスト用データ-ラベル

#################################################
# 学習用データ読込
#################################################
def read_train_image():
  
    num_class = len(TRAIN_IMAGE_DIRS)

    train_images = []
    train_labels = []

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
                print(filepath)
                continue

            # 画像をモデルの入力サイズにリサイズする。
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.flatten().astype(numpy.float32) / 255.0
            #simage = image.flatten()
            train_images.append(image)

            # 学習データに対応するラベルを作成する。
            label = numpy.zeros(num_class)
            label[i] = 1.0
            train_labels.append(label)

    return(train_images, train_labels)


#################################################
# 学習用データ読込
#################################################
def resize_train_image():

    num_class = len(TRAIN_IMAGE_DIRS)

    # 学習データディレクトリから画像を読込む
    for i, dir in enumerate(TRAIN_IMAGE_DIRS):

        train_image_dir = os.path.join(TRAIN_IMAGE_ROOTDIR, dir)
        files = os.listdir(train_image_dir)

        resized_image_dir = os.path.join(TRAIN_IMAGE_ROOTDIR, RESIZED_IMAGE_DIRS[i])
        if os.path.isdir(resized_image_dir):
            shutil.rmtree(resized_image_dir)
        os.mkdir(resized_image_dir)

        # TRAIN_IMAGE_DIRS内の全画像を読込み、
        # 画像のリサイズとラベルの作成を行う。
        for filename in files:

            # 画像を読み込む。
            input_filepath = os.path.join(train_image_dir, filename)
            image = cv2.imread(input_filepath)
            if image is None:
                print(input_filepath)
                continue

            # 画像をモデルの入力サイズにリサイズする。
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

            # リサイズした画像を書き込む。
            output_filepath = os.path.join(resized_image_dir, filename)
            cv2.imwrite(output_filepath, image)

    return


#################################################
# テスト用データ読込
#################################################
def read_test_image():
    num_class = len(TEST_IMAGE_DIRS)

    test_images = []
    test_labels = []

    # 学習データディレクトリから画像を読込む
    for i, dir in enumerate(TEST_IMAGE_DIRS):
        test_image_dir = os.path.join(TEST_IMAGE_ROOTDIR, dir)
        files = os.listdir(test_image_dir)

        # TEST_IMAGE_DIRS内の全画像を読込み、
        # 画像のリサイズとラベルの作成を行う。
        for filename in files:

            # 画像を読み込む。
            filepath = os.path.join(test_image_dir, filename)
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

    return(test_images, test_labels)

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

        # dogの画像をダウンロード
        num_dog_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'dog'), 'n02084071', 2000)
        print('%d images of dog are downloaded.' % num_dog_image )

        # catの画像をダウンロード
        num_cat_image = downloader.download_image(os.path.join(TRAIN_IMAGE_ROOTDIR, 'cat'), 'n02121808', 2000)
        print('%d images of cat are downloaded.' % num_cat_image)

        # テスト用画像はBingからダウンロードしました

    elif argvs[1] == 'train':
         
        # 学習用データ読込み
        print('--- Read_train image ---')
        #train_images, train_labels = read_train_image()

        #print('  %d images read.' % len(train_images))

        # 入力画像をZCA白色化する
        # ZCA白色化したら逆に精度が落ちたのでコメントアウト
        #print('--- zca_whitening ---')
        #train_images = image_editor.zca_whitening(train_images)
        #print(train_images.shape)

        # 入力画像に前処理を施す
        #print('--- Preprocessing image ---')
        #train_images, train_labels = image_editor.preprocessing_images(train_images, train_labels, RATE_INCREASE)

        resize_train_image()

        dirs = []
        for dir in RESIZED_IMAGE_DIRS:
            train_image_dir = os.path.join(TRAIN_IMAGE_ROOTDIR, dir)
            dirs.append(train_image_dir)

        train_images, train_labels = image_editor.input_distorted_images(dirs, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, RATE_INCREASE)

        #tmp_image = numpy.resize(train_images[0], (IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS))
        #tmp_image = tmp_image.astype(numpy.uint8)
        #cv2.imshow('Test', tmp_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # モデル生成
        print('--- Create model ---')
        num_class = len(TRAIN_IMAGE_DIRS)
        model = mymodel.mymodel(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, num_class, CKPT_DIR, MODEL_FILENAME)
        model.construct()

        # モデル実行
        print('--- Training start ---')
        #model.train(train_images, train_labels, STEPS, BATCH_SIZE)
        model.train(train_images, train_labels, STEPS, BATCH_SIZE, SAVE_STEP)
        #model.train(train_images.tolist(), train_labels, STEPS, BATCH_SIZE)

        # モデル保存
        print('--- Save model ---')
        model.save()

    elif argvs[1] == 'test':

        # テスト用データ読込み
        print('--- Read_test image ---')
        test_images, test_labels = read_test_image()
        print('  %d images read.' % len(test_images))

        # 入力画像をZCA白色化する
        #print('zca_whitening')
        #test_images = image_editor.zca_whitening(test_images)

        # モデル生成
        print('--- Restore model ---')
        num_class = len(TEST_IMAGE_DIRS)
        model = mymodel.mymodel(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, num_class, CKPT_DIR, MODEL_FILENAME)
        model.load(CKPT_DIR, MODEL_FILENAME)

        # モデル実行
        print('--- Test start ---')
        model.test(test_images, test_labels)
        #model.test(test_images.tolist(), test_labels, BATCH_SIZE)

    elif argvs[1] == 'face':

        #img_path =  './train_data/dog/80ec119a-67c0-4da1-b072-1b939ad6345e.jpg'
        img_path =  './train_data/cat/013gatto_t.jpg'
        #img_path =  './train_data/cat/00.jpg'
        #img_path =  './train_data/dog/2.jpg'

        img = cv2.imread(img_path)
        #image_editor.detect_face(img)
        cv2.imshow('Test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print_usage(argvs[0])
        quit()
        
# EOF
