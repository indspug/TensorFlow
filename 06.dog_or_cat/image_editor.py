# -*- coding: utf-8 -*-

# image_editor.py

# import
import os
import cv2
import numpy
import csv
import tensorflow as tf

#################################################
# 指定したファイルパスの画像を読込み返す。
# ラベルはディレクトリ単位で付加する。
# [Args]:
#    image_filepaths :読込む画像が格納されたディレクトリのリスト
# [Returns]:
#    images:画像のリスト
#           (numpy.ndarray形式のリスト,RGBの並びはOpenCV用)
#    labels:ラベルのリスト
#           (numpy.ndarray形式のリスト,RGBの並びはOpenCV用)
#################################################
def get_images(image_filepaths):

    images = []

    # 指定したファイルパスの画像を読み込む。
    for image_filpath in image_filepaths:

        image = cv2.imread(image_filpath)
        if image is None:
            continue

        # 画像をリストに追加
        images.append(image)


    return(images)


#################################################
# 指定ディレクトリに格納された画像と画像のラベルを返す
# ラベルはディレクトリ単位で付加する。
# [Args]:
#    image_dirs:読込む画像が格納されたディレクトリのリスト
# [Returns]:
#    images:画像のリスト
#           (numpy.ndarray形式のリスト,RGBの並びはOpenCV用)
#    labels:ラベルのリスト
#           (numpy.ndarray形式のリスト,RGBの並びはOpenCV用)
#################################################
def get_labeled_images(image_dirs):

    images = []
    labels = []

    num_class = len(image_dirs)

    # 指定したディレクトリ群内の画像を読込む
    for i, image_dir in enumerate(image_dirs):

        files = os.listdir(image_dir)

        # image_dir内の全画像を読込む。
        for file in files:

            # 画像を読み込む。
            filepath = os.path.join(image_dir, file)
            image = cv2.imread(filepath)
            if image is None:
                continue

            # 画像をリストに追加
            images.append(image)

            # ラベルを作成する。
            label = numpy.zeros(num_class)
            label[i] = 1.0
            labels.append(label)

    return(images, labels)

#################################################
# リスト内の画像を指定したサイズにリサイズして返す。
# [Args]:
#    images :画像のリスト(numpy.ndarrayのリスト,RGBの並びはOpenCV用)
#    width  :リサイズ後の幅
#    height :リサイズ後の高さ
# [Returns]:
#    resized_images :リサイズした画像のリスト
#           (numpy.ndarrayのリスト,RGBの並びはOpenCV用)
#################################################
def resize_images(images, width, height):

    resized_images = []

    for image in images:
        resized_image = cv2.resize(image, (width, height))
        resized_image = resized_image.flatten().astype(numpy.float32) / 255.0
        resized_images.append(resized_image)

    return(resized_images)

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
    record_defaults = [["s"]]
    record_defaults.extend(label_form)

    # csvからファイル名とラベル取出し
    row = tf.decode_csv(val, record_defaults=record_defaults)
    fname = row[0]
    label = row[1:]
    label = tf.cast(label, tf.float32)

    # JPEG読込み
    jpeg_r = tf.read_file(fname)
    image = tf.image.decode_jpeg(jpeg_r, channels=color_channels)

    # JPEG画像を指定サイズにリサイズ
    image = tf.cast(image, tf.float32)
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