# -*- coding: utf-8 -*-

# import
import os
import csv
import numpy
import cv2
import tensorflow as tf

#################################################
# 入力画像を指定したサイズにリサイズする。
#################################################
#def resize_image(image, width, height):
#    return(cv2.resize(image, (width, height)))

#################################################
# 入力配列を[ [...] ]の形に変換する。
#################################################
def flatten_matrix(matrix):
    vector = matrix.flatten()
    vector = vector.reshape(1, len(vector))
    return(vector)

#################################################
# 入力画像(numpy配列)に対してZCA白色化を行う
#################################################
def zca_whitening(input_images):
    
    # 全画像に対して正規化を行う
    count = 0
    image_matrix = None
    for image in input_images:
        mean = numpy.mean(image)    # 平均値
        var = numpy.var(image)      # 分散
        meaned_image = (image-mean) / float(numpy.sqrt(var))
        flatten_image = flatten_matrix(meaned_image)
        if count == 0:
            image_matrix = numpy.array(flatten_image)
        else:
            image_matrix = numpy.r_[image_matrix, flatten_image]

        count += 1

    # ZCA白色化の変換行列を作成する。
    num_images = float(image_matrix.shape[0])
    sigma = numpy.dot(image_matrix.T, image_matrix)/num_images  # 共分散行列
    U,S,V = numpy.linalg.svd(sigma)                             # 特異値分解
    epsilon = 0.01 # 0割防止のため0より大きい値にしている
    sqrt_S = numpy.sqrt(S + epsilon)
    #zca_matrix = numpy.dot(numpy.dot(U, numpy.diag(1.0/sqrt_S)), U.T)
    zca_matrix = numpy.dot(U / sqrt_S[numpy.newaxis, :], U.T)
    
    return numpy.dot(image_matrix, zca_matrix.T) # ZCA白色化を行った画像を返す

#################################################
# 入力画像から顔検出を行う
#################################################
def detect_face(input_image):

    print('detect_face')

    # 分類器の選択
    #classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #classifier = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    classifier = cv2.CascadeClassifier('./train_data/cascade.xml')

    #print(classifier)

    # 画像をグレーに変換
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # 入力画像か顔検出
    #faces = classifier.detectMultiScale(gray_image, 1.1)
    faces = classifier.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 1)

    # デバッグ用
    for face in faces:
        print(face)
        xy1 = tuple(face[0:2])
        xy2 = tuple(face[0:2] + face[2:4])
        cv2.rectangle(gray_image, xy1, xy2, (255,255,255),thickness=2)
    
    cv2.imshow('TEST', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#################################################
# 入力画像に前処理を行う
#################################################
def preprocessing_images(input_images, input_labels, rate_increase):

    output_images = []
    output_labels = []
    count = 0

    image_variables = []

    # セッション生成
    session = tf.Session()
    #session = tf.InteractiveSession()

    for i, image in enumerate(input_images):

        for j in range(rate_increase):

            # ランダムに左右反転
            fliped_image = tf.image.random_flip_left_right(image)

            # 輝度をランダムに変える
            bright_changed_image = tf.image.random_brightness(fliped_image, max_delta=63)

            # コントラストをランダムに変える
            contrast_changed_image = tf.image.random_contrast(bright_changed_image, lower=0.2, upper=1.8)

            # 画像を白色化する
            whiten_image = tf.image.per_image_whitening(contrast_changed_image)
            whiten_image = session.run(whiten_image)

            #whiten_image = session.run(fliped_image)
            #whiten_image = fliped_image.eval()

            # リストに画像を追加する
            output_images.append(whiten_image.flatten())

            image_variables.append(whiten_image)

            # ラベルをコピーする
            label = input_labels[i].copy()

            # リストにラベルを追加する
            output_labels.append(label)


        count += 1
        print(count)


    # セッション終了
    session.close()

    return(output_images, output_labels)

#################################################
# 入力画像に前処理を行う
#################################################
def input_distorted_images(input_dirs, image_width, image_height, color_channels, rate_increase):

    # ファイル名の一覧取得
    num_labels = len(input_dirs)

    # CSVファイルを開く
    csv_name = './filenames.csv'
    fout = open(csv_name, 'w')
    csvWriter = csv.writer(fout)

    # CSVにファイルパスと画像の幅,高さ、ラベルの一覧書き込み
    num_files = 0
    for i, input_dir in enumerate(input_dirs):

        # ファイル一覧取得
        files = os.listdir(input_dir)
        for file in files:

            # ファイルパス作成
            filepath = os.path.join(input_dir, file)

            # ラベル作成
            label = ['0'] * num_labels
            label[i] = '1'

            str_row = [filepath]
            str_row.extend(label)

            # 1枚の画像からN枚の画像を作るために同じデータをN回書き込む
            for n in range(rate_increase):
                csvWriter.writerow(str_row)

            num_files += 1

    fout.close()

    # csvからファイルパスとラベル読込み
    filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
    reader = tf.TextLineReader()
    key, val = reader.read(filename_queue)

    # CSVから1行取出すときのフォーマット( [文字列(ファイル名)], [0] × ラベルの数  )
    label_form = [[0 for i in range(1)] for j in range(num_labels)]
    #record_defaults = [["s"], [0], [0]]
    record_defaults = [["s"]]
    record_defaults.extend(label_form)

    # csvからファイル名とラベル取出し
    row = tf.decode_csv(val, record_defaults=record_defaults)
    fname = row[0]
    #width = row[1]
    #height = row[2]
    label = row[1:]
    label = tf.cast(label, tf.float32)

    # JPEG読込み
    jpeg_r = tf.read_file(fname)
    image = tf.image.decode_jpeg(jpeg_r, channels=color_channels)

    # JPEG画像を指定サイズにリサイズ
    image = tf.cast(image, tf.float32)
    #image.set_shape([width, height, color_channels])
    image.set_shape([image_width, image_height, color_channels])
    image = tf.image.resize_image_with_crop_or_pad(image, image_width, image_height)

    # ランダムに左右反転
    fliped_image = tf.image.random_flip_left_right(image)

    # 輝度をランダムに変える
    bright_changed_image = tf.image.random_brightness(fliped_image, max_delta=63)

    # コントラストをランダムに変える
    contrast_changed_image = tf.image.random_contrast(bright_changed_image, lower=0.2, upper=1.8)

    # 画像を白色化する
    whiten_image = tf.image.per_image_whitening(contrast_changed_image)
    #whiten_image = tf.image.per_image_whitening(image)

    # JPEG画像を指定サイズにリサイズ
    #image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
    #image = tf.image.resize_image_with_crop_or_pad(whiten_image, image_height, image_width)

    # queueを実行する準備
    sess = tf.Session()
    #init = tf.initialize_all_variables()
    #sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # queueを使用し並列にファイル読込み〜学習データ取得を行う
    train_images = []
    train_labels = []
    num_images = num_files * rate_increase
    try:
        for i in range(num_images):
            if coord.should_stop():
                break
            #train_image = sess.run(contrast_changed_image)
            train_image = sess.run(whiten_image)
            train_images.append(train_image.flatten())
            train_label = sess.run(label)
            train_labels.append(train_label)

    except tf.errors.OutOfRangeError:
        print("OutOfRangeError")

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    return (train_images, train_labels)