# YOLOv2（Chainerバージョン）
本稿は、YOLOv2の論文をChainer上で再現実装したものです。darknetオリジナルの学習済みパラメータファイルをchainerで読み込むためのパーサと、chainer上で完全にゼロからYOLOv2を訓練するための実装が含まれています。


Joseph Redmonさんの論文はこちら：

[You Only Look Once](https://arxiv.org/abs/1506.02640)

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)


darknetのオリジナルはこちら：

[darknet](http://pjreddie.com/)

chainerでYOLOv2(tiny版のみ)の読み込む方法についてはこちら：

[YOLOv2でPPAP](http://qiita.com/ashitani/items/566cf9234682cb5f2d60)




<img src="data/dance_short.gif">

<img src="data/drive_short.gif">

<img src="data/animal_output.gif">


## 環境
- Ubuntu 16.04.1 LTS (GNU/Linux 4.4.0-59-generic x86_64)
- Anaconda 2.4.1
- Python 3.5.2
- OpenCV 3.1.0
- Chainer 1.17.0
- CUDA V8.0


## 訓練済みYOLOv2モデル(完全版)の実行
darknetオリジナルの重みパラメータファイルをchainerで読み込んで実行するための手順です。<a href="http://qiita.com/ashitani/items/566cf9234682cb5f2d60">こちら</a>を参考にさせて頂きました。

１、yolov2学習済みweightsファイルをダウンロードする。


```
wget http://pjreddie.com/media/files/yolo.weights
```

２、以下のコマンドでweightsファイルをchainer用にパースする。

```
python yolov2_darknet_parser.py yolo.weights
```

３、以下のコマンドで好きな画像ファイルを指定して物体検出を行う。
検出結果は`yolov2_result.jpg`に保存される。

```
python yolov2_darknet_predict.py data/people.png
```


４、以下のコマンドで、カメラを起動しリアルタイム物体検出を行う。

```
python yolov2_darknet_camera.py 
```


## Chainer上でYOLOv2の訓練
フリー素材の動物アイコンデータセットを使ったYOLOv2の訓練です。長くなるので、別ページにまとめました。

<a href="./YOLOv2_animal_train.md">YOLOv2を使った動物アイコンデータセットの訓練手順</a>

こちらをご覧ください。


## YOLOv2の理論解説
YOLOv2の論文及びdarknetオリジナルの実装についての解説です。こちらも長くなるので別ページにまとめました。

<a href="./YOLOv2.md">YOLOv2の仕組み解説</a>

こちらをご覧ください。