# -*- coding: utf-8 -*-

# image_editor.py

# import
import os
import cv2
import numpy

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