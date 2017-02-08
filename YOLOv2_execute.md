# YOLOv2の訓練済みモデル実行手順
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