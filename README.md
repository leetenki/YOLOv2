# YOLOv2（Chainerバージョン）
YOLOv2は、2016年12月25日時点の、速度、精度ともに世界最高のリアルタイム物体検出手法です。

本リポジトリは、YOLOv2の論文をChainer上で再現実装したものです。darknetオリジナルの学習済みパラメータファイルをchainerで読み込むためのパーサと、chainer上でゼロからYOLOv2を訓練するための実装が含まれています。（YOLOv2のtiny版に関してはChainerで読み込む方法が<a href="http://qiita.com/ashitani/items/566cf9234682cb5f2d60">こちら</a>のPPAPの記事で紹介されています。今回は、Full Version のYOLOv2の読込みと、学習ともにChainer実装しています。）



Joseph Redmonさんの元論文はこちら：

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016/12/25)


[You Only Look Once](https://arxiv.org/abs/1506.02640) 


darknetのオリジナル実装はこちら：

[darknet](http://pjreddie.com/)

chainerを使ったYOLOv2(tiny版)の読み込む方法についてはこちら：

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



## YOLOv2の訓練済みモデル(完全版)実行手順
darknetオリジナルの重みパラメータファイルをchainerで読み込んで実行するための手順です。

<a href="./YOLOv2_execute.md">訓練済みYOLOの実行手順</a>

こちらのページにまとめました。


## YOLOv2の訓練手順
フリー素材の動物アイコンデータセットを使ったYOLOv2の訓練です。

<a href="./YOLOv2_animal_train.md">YOLOv2を使った動物アイコンデータセットの訓練手順</a>

こちらのページに手順をまとめました。



## YOLOv2の理論解説
YOLOv2の論文及びdarknetオリジナルの実装についての解説です。こちらも別ページにまとめました。

<a href="./YOLOv2.md">YOLOv2の仕組み解説</a>