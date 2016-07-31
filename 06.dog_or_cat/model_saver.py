# -*- coding: utf-8 -*-

# model_saver.py

# import
import tensorflow as tf
import os

#################################################
# モデルの保存
# [Args]:
#    session         :tf.Session
#    variables       :保存するVariables([tf.Variables, ...])
#    checkpoint_dir  :チェックポイントを保存するディレクトリ
#    checkpoint_name :保存するチェックポイントのファイル名
#    global_step     :学習ステップ(途中経過を保存する際に指定する)
# [Returns]:
#    saved_path:保存されたチェックポイントのファイルパス
#################################################
def save_model(session, variables, checkpoint_dir, checkpoint_name, global_step=0):

    # ディレクトリ作成
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, checkpoint_name)
    saver = tf.train.Saver(variables)

    if global_step == 0:
        saved_path = saver.save(session, filepath)
    else:
        saved_path = saver.save(session, filepath, global_step=global_step)

    return(saved_path)

#################################################
# モデルの読込
# [Args]:
#    session:tf.Session
#    checkpoint_dir:読込むチェックポイントが格納されているディレクトリ
#    checkpoint_name:読込むチェックポイントのファイル名
#################################################
def restore_model(session, checkpoint_dir, filename):

    # チェックポイントの存在確認
    if exists_checkpoint(checkpoint_dir, filename):
        filepath = os.path.join(checkpoint_dir, filename)
        saver = tf.train.Saver()
        saver.restore(session, filepath)
    else:
        filepath = os.path.join(checkpoint_dir, filename)
        message = 'Can\'t open ' + filepath
        raise IOError(message)

#################################################
# モデルの存在確認
# [Args]:
#    checkpoint_dir:読込むチェックポイントが格納されているディレクトリ
#    checkpoint_name:読込むチェックポイントのファイル名
# [Returns]:
#    exists:チェックポイントが存在する場合はTrue,それ以外はFalse
#################################################
def exists_checkpoint(checkpoint_dir, filename):

    filepath = os.path.join(checkpoint_dir, filename)

    # 指定したファイルパスがチェックポイントに格納されているか検索する。
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    exists = False
    for read_filename in ckpt.all_model_checkpoint_paths:
        if read_filename == filepath:
            exists = True
            break

    return(exists)