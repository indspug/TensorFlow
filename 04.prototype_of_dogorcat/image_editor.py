# -*- coding: utf-8 -*-

# import
import numpy
import cv2

#################################################
# 入力画像を指定したサイズにリサイズする。
#################################################
def resize_image(image, width, height):
    return(cv2.resize(image, (width, height)))

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
    zca_matrix = numpy.dot(numpy.dot(U, numpy.diag(1.0/sqrt_S)), U.T)
    
    return numpy.dot(image_matrix, zca_matrix.T) # ZCA白色化を行った画像を返す
