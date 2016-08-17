# -*- coding: utf-8 -*-

# test_main.py

# import
import os
import sys
import csv
import image_editor
from dog_or_cat_model import DogOrCatModel

# 定数
IMAGE_WIDTH = 30    # 入力画像の幅
IMAGE_HEIGHT = 30   # 入力画像の高さ
COLOR_CHANNELS = 3  # カラーチャンネルの数
NUM_CLASSES = 2     # 正解(ラベル)の数(種類)
BATCH_SIZE = 20     # バッチサイズ
STEPS = 1000        # 学習ステップ
SAVE_STEP = 50      # 一定ステップごとにモデルを保存する

#################################################
# ログ出力
# [Args]:
#    message:ログに出力するメッセージ
#################################################
def write_log(message):
    log = '--- ' + message + ' ---'
    print(log)

#################################################
# Usage出力
# [Args]:
#    exe_name :実行ファイル名
#################################################
def print_usage(exe_name):
    print('Usage: # python %s train [checkpoint] [image_dir] [continuation] [start_step]' % exe_name)
    print('Usage: # python %s test  [checkpoint] [image_dir]' % exe_name)
    print('Usage: # python %s demo  [checkpoint] [imagefile] [resultfile]' % exe_name)
    print('  checkpoint   : filepath of checkpoint.')
    print('  image_dir    : root directory of image directories.')
    print('  continuation : whether to continue training or not (optional).')
    print('  start_step   : number of start step. only needed when continuation set (optional).')
    print('  imagefile    : filepath of imagefile for demo.')
    print('  resultfile   : demo result filepath.')


#################################################
# 学習実行
# [Args]:
#    checkpoint_path     :チェックポイントのファイルパス
#    train_image_rootdir :学習画像格納ディレクトリ群のルートディレクトリ
#    continuation        :学習を途中から再開するか否か
#    start_step          :学習開始のステップ
#################################################
def train(checkpoint_path, train_image_rootdir, continuation, start_step):

    # 画像ファイルの読み込みとリサイズ
    write_log('Read image files')
    train_image_dirs = []
    for image_dir in os.listdir(train_image_rootdir):
        dir = os.path.join(train_image_rootdir, image_dir)
        if os.path.isdir(dir):
            train_image_dirs.append(dir)

    images, labels = image_editor.get_labeled_images(train_image_dirs)
    images = image_editor.resize_images(images, IMAGE_WIDTH, IMAGE_HEIGHT)

    # モデルオブジェクトの構築
    model = DogOrCatModel(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, NUM_CLASSES, checkpoint_path)
    if (continuation):
        write_log('Restore model')
        model.load()
        model.ready_for_train()
    else:
        write_log('Construct model')
        model.construct()

    # 学習の実行
    write_log('Start training')
    model.train(images, labels, STEPS, BATCH_SIZE, SAVE_STEP, start_step)

    # モデル保存
    write_log('Save model')
    model.save()

#################################################
# テスト実行
# [Args]:
#    checkpoint_path    :チェックポイントのファイルパス
#    test_image_rootdir :テスト画像格納ディレクトリ群のルートディレクトリ
#################################################
def test(checkpoint_path, test_image_rootdir):

    # 画像ファイルの読み込みとリサイズ
    write_log('Read image files')
    test_image_dirs = []
    for image_dir in os.listdir(test_image_rootdir):
        dir = os.path.join(test_image_rootdir, image_dir)
        if os.path.isdir(dir):
            test_image_dirs.append(dir)

    images, labels = image_editor.get_labeled_images(test_image_dirs)
    images = image_editor.resize_images(images, IMAGE_WIDTH, IMAGE_HEIGHT)

    # モデルオブジェクトのロード
    write_log('Load model')
    model = DogOrCatModel(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, NUM_CLASSES, checkpoint_path)
    model.load()

    # テストの実行
    write_log('Start test')
    test_accuracy = model.test(images, labels)

    # 正解率の表示
    print("test accuracy = %g" % test_accuracy)


#################################################
# デモ実行
# [Args]:
#    checkpoint_path :チェックポイントのファイルパス
#    imagefile_path  :デモ用画像のファイルパス
#    resultfile_path :識別結果出力先ファイルパス
#################################################
def demo(checkpoint_path, imagefile_path, resultfile_path):

    # デモ用画像の読込みとリサイズ
    images = image_editor.get_images([imagefile_path])
    images = image_editor.resize_images(images, IMAGE_WIDTH, IMAGE_HEIGHT)

    # モデルオブジェクトのロード
    write_log('Load model')
    model = DogOrCatModel(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS, NUM_CLASSES, checkpoint_path)
    model.load()

    # 学習の実行
    write_log('Start demo')
    scores = model.get_scores(images)

    # スコアの表示
    print('Score')
    for i, score in enumerate(scores):
        print('  %02d :' % i),
        for j, rate in enumerate(score):
            if j==0:
                print('cat=%.1f%%,' % (rate*100)),
            elif j == 1:
                print('dog=%.1f%%'  % (rate*100))

    # スコアをCSVに出力
    fout = open(resultfile_path, 'w')
    csvWriter = csv.writer(fout)
    csvWriter.writerows(scores)
    fout.close()

#################################################
# エントリーポイント
# [Args]:
#################################################
if __name__ == '__main__':

    # コマンドライン引数のチェック
    write_log('Check arguments')
    argvs = sys.argv
    argc = len(argvs)
    if argc < 2:
        print_usage(argvs[0])
        quit()

    mode = argvs[1]

    # 学習用モード
    if mode == 'train':

        # コマンドライン引数取得
        # 引数が[checkpoint] [image_dir]ならOK
        if(argc == 4):
            continuation = False
            start_step = 0
        # 引数が[checkpoint] [image_dir] [continuation][start_step]ならOK
        elif (argc == 6):
            start_step = int(argvs[5])
            if (argvs[4].lower() == 'false'):
                continuation = False
            elif (argvs[4].lower() == 'true'):
                continuation = True
            else:
                print_usage(argvs[0])
                quit()
        else:
            print_usage(argvs[0])
            quit()

        checkpoint_path = argvs[2]
        train_image_rootdir = argvs[3]

        train(checkpoint_path, train_image_rootdir, continuation, start_step)

    # テスト用モード
    elif mode == 'test':

        # コマンドライン引数取得
        if argc < 4:
            print_usage(argvs[0])
            quit()

        checkpoint_path = argvs[2]
        train_image_rootdir = argvs[3]

        test(checkpoint_path, train_image_rootdir)

    # デモ用モード
    elif mode == 'demo':

        # コマンドライン引数取得
        if argc < 5:
            print_usage(argvs[0])
            quit()

        checkpoint_path = argvs[2]
        imagefile_path = argvs[3]
        resultfile_path = argvs[4]

        demo(checkpoint_path, imagefile_path, resultfile_path)

    else:
        print_usage(argvs[0])
        quit()

# EOF
