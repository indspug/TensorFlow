# -*- coding: utf-8 -*-

# import
import os
import subprocess
from urllib2 import urlopen
import imghdr
import random

# 定数
URL_TEXT='url.txt'
GETURL_URL='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
SIZE_FLICKR_ERROR = 2051

#################################################
# imagenetから指定したsynnet IDの画像を
# 指定枚数(num_get_files)取得し、
# 指定したディレクトリ(dir)に格納する。
#################################################
def download_image(dir, synnet_id, num_get_files):
  
  # ディレクトリ作成
  if not os.path.isdir(dir):
    os.mkdir(dir)

  # 指定したsynnet IDに対応する画像のURLリスト取得
  url_text_path = dir + '/' + URL_TEXT
  url = GETURL_URL + synnet_id
  urls = urlopen(url).read().split()
  if len(urls) <= 0:
    print('failed to get urls of image!')
    return
  
  # 取得したURLから画像をダウンロード
  actual_num_files = 0
  for url in urls:

    splited_url = url.split('/')               # '/'で分割
    filename = splited_url[len(splited_url)-1] # ファイル名取出し
    filepath = dir + '/' + filename
    
    # 既に同名ファイルが存在する場合はダウンロードしない
    if os.path.exists(filepath):
      continue

    # ファイルが存在しない場合はダウンロード
    image_dl_cmd = 'wget -t 1 -O' + filepath + ' ' + url
    try:
      status = subprocess.check_call(image_dl_cmd.split(' '))
    except subprocess.CalledProcessError, e:
      status = e.returncode
 
    # ダウンロードに成功したか確認
    if status == 0:
       
      # ダウンロードした画像がFlickr Errorでないか確認
      size = os.path.getsize(filepath)
      # ダウンロードした画像のフォーマット確認
      ext = imghdr.what(filepath)
      if size == SIZE_FLICKR_ERROR: 
        # Flickr Errorの場合はファイル削除
        os.remove(filepath)
      elif ext == None:
        # 画像のフォーマットが異常な場合はファイル削除
        os.remove(filepath)
      else:
        # Error無しの場合は画像枚数加算
        actual_num_files += 1  
        print(filename)
    else:
      os.remove(filepath)
    
    #print([status, actual_num_files])

    # 指定された枚数をダウンロードしたら終了
    if actual_num_files >= num_get_files:
      break 
  
  # 実際に取得できた画像の枚数を返す
  return(actual_num_files)
