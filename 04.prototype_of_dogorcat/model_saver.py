# -*- coding: utf-8 -*-

# import
import tensorflow as tf
import os

#################################################
# モデルの保存
#################################################
def save_model(session, variables, checkpoint_dir, filename):

    # ディレクトリ作成
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, filename)
    saver = tf.train.Saver(variables)
    saved_path = saver.save(session, filepath)
    return(saved_path)

#################################################
# モデルの読込
#################################################
def restore_model(session, checkpoint_dir, filename):
    if exists_model(checkpoint_dir, filename):
        filepath = os.path.join(checkpoint_dir, filename)
        saver = tf.train.Saver()
        saver.restore(session, filepath)
    else:
        filepath = os.path.join(checkpoint_dir, filename)
        message = 'Can\'t open ' + filepath
        raise IOError(message)

#################################################
# モデルの存在確認
#################################################
def exists_model(checkpoint_dir, filename):

    filepath = os.path.join(checkpoint_dir, filename)

    # 指定したファイルパスがチェックポイントに格納されているか
    # 検索する。
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    exists = False
    for read_filename in ckpt.all_model_checkpoint_paths:
        if read_filename == filepath:
            exists = True
            break

    return(exists)

