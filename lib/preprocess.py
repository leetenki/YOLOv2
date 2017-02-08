import argparse
import urllib.request

def download_image(): 
    parser = argparse.ArgumentParser(description='Web上から画像をダウンロードし、処理を行う。')
    parser.add_argument('--url', '-u', default='http://blog-imgs-63.fc2.com/o/o/i/ooinarukuu/20140502104641477.jpg', help='ダウンロードするイメージのURLを指定する')
    args = parser.parse_args()

    print('Download Image From {0} ....'.format(args.url))
    image_file_path = './sample_images/sample.jpg'
    urllib.request.urlretrieve(args.url, image_file_path)

    return image_file_path
