# -*- coding: utf-8 -*-

# import
import tensorflow as tf
import os

#################################################
# モデルの保存
#################################################
def save_model(session, variables, checkpoint_dir, filename):
  
  filepath = os.join(checkpoint_dir, filename)
  saver = tf.traiin.Saver(variables)
  saved_path = saver.save(session, filepath)
  return(saved_path)

#################################################
# モデルの読込
#################################################
def restore_model(session, checkpoint_dir, filename):
  if exists_model(checkpoint_dir, filename):
    filepath = os.join(checkpoint_dir, filename)
    saver = tf.traiin.Saver()
    saver.restore(session, filepath)
  else:
    filepath = os.join(checkpoint_dir, filename)
    message = 'Can\'t open ' + filepath
    raise IOError(message)

#################################################
# モデルの存在確認
#################################################
def exists_model(checkpoint_dir, filename):
  ckpt = tf.train.get_check_point_state(checkpoint_dir)
  
  exists = False
  for read_filename in ckpt.all_model_checkpoint_paths:
    if read_filename == filename:
      exists = True
      break

  return(exists)

