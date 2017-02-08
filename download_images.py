import urllib.request

items = [
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur01_b_01.png", "./items/Tyrannosaurus.png"],
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur01_b_02.png", "./items/Triceratops.png"],
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur01_b_03.png", "./items/Ammonite.png"],
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur02_b_01.png", "./items/Erasmosaurus.png"],
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur02_b_02.png", "./items/Brachiosaurus.png"],
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur03_b_01.png", "./items/Riopururodon.png"],
    ["http://putiya.com/img/animal/dinosaur/dinosaur01/dinosaur03_b_02.png", "./items/Parasarolophus.png"],
    ["http://putiya.com/img/animal/dog_banken/dog01_banken/dog01_banken__b_05.png", "./items/Dog.png"],
    ["http://putiya.com/img/animal/animal_english/animal/english_animal_b_10.png", "./items/Deer.png"],
    ["http://putiya.com/img/animal/animal_english/animal/english_animal_b_17.png", "./items/Rhino.png"]
]

backgrounds = [
    ["http://www.priga.jp/imgdl/DL00202.jpg", "./backgrounds/0.jpg"],
    ["http://www.priga.jp/imgdl/DL00201.jpg", "./backgrounds/1.jpg"],
    ["http://www.priga.jp/imgdl/DL00101.jpg", "./backgrounds/2.jpg"],
    ["http://www.priga.jp/imgdl/DL00099.jpg", "./backgrounds/3.jpg"],
    ["http://www.priga.jp/imgdl/DL00098.jpg", "./backgrounds/4.jpg"],
    ["http://www.priga.jp/imgdl/DL00095.jpg", "./backgrounds/5.jpg"],
    ["http://www.priga.jp/imgdl/DL00082.jpg", "./backgrounds/6.jpg"],
]

for item in items:
    urllib.request.urlretrieve(item[0], item[1])

for background in backgrounds:
    urllib.request.urlretrieve(background[0], background[1])
