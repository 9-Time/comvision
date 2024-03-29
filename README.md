# Computer Vision Project
SUTD 50.035 Computer Vision Course Project

Security object detection

Model has been trained to look for `Handgun,Knife,Beer,Baseball bat,Skateboard`

## Getting Started
*YOLO Variant was just a test and not implemented in the end, therefore has no training or evaluation method

For SSD, to train the model,

Ensure that data is in the form of
```
.
+-- open_images/
|   +-- test/
|       +-- <testimages>
|   +-- train/
|       +-- <trainimages>
|   +-- validation/
|       +-- <validationimages>
|   +-- class-descriptions-boxable.csv
|   +-- sub-test-annotations-bbox.csv
|   +-- sub-train-annotations-bbox.csv
|   +-- sub-validation-annotations-bbox.csv
|   +-- test-annotations-bbox.csv
|   +-- train-annotations-bbox.csv
|   +-- validation-annotations-bbox.csv
```
You may use the given script in the file to download it (Script from https://github.com/qfgaohao/pytorch-ssd). and example on how to run it is as such: `python open_images_downloader.py --root ./data/open_images --class_names "Handgun,Knife,Beer,Baseball bat,Skateboard"`

Edit the hyper parameters in `train.py` and run the file

Next, to eval, run `eval.py` (similarly after changing hyperparmeters)

Finally, you can draw boundingboxes on your eval using `draweval.py`, be sure to edit the respective file path

# Live Camera Detection

## YOLO Variant (securitybyyolo)
Requires the following

```
$ pip install python-opencv
$ pip install torch
$ pip install torchvision
$ pip install numpy
```
To run the livefeed with detection
```
$ git clone https://github.com/scarmaten/comvision.git
$ cd comvision/securitybyyolo
$ python livedarknet.py
```
Press q to quit

## SSD Variant (securitybyssd)
Requires the following

```
$ pip install python-opencv
$ pip install torch
$ pip install torchvision
$ pip install numpy
```

To download our model, head over to
https://www.dropbox.com/s/lv5ye9u0k7zpjvy/vgg-Epoch-10-Loss-3.0180892944335938.pth?dl=0

To download our version 2 model (using kernel size 1 and 0 padding on last layer), head over to 
https://www.dropbox.com/s/9j7mf9tvi01i10j/v2-Epoch-39-Loss-7.86013218334743.pth?dl=0

Version 2 works better from our testing

### Differences Between Version 1 and Version 2 Model
Version 1
```
self.classification_headers = ModuleList([
    Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), 
])

self.regression_headers = ModuleList([
    Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), 
])
```

Version 2
```
self.classification_headers = ModuleList([
    Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=1, padding=0), 
])

self.regression_headers = ModuleList([
    Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
    Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=1, padding=0), 
])
```

### To run the livefeed with detection
```
$ git clone https://github.com/scarmaten/comvision.git
$ cd comvision/securitybyssd
$ python livedetection.py
```
Press q to quit

# Authors
* Kah Wee
* Rachel
* Venessa
* Jun Qing

# Acknowledgements
This project referenced and adapted various sources

* https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
* https://github.com/qfgaohao/pytorch-ssd
* https://github.com/amdegroot/ssd.pytorch

Dataset used for training from OpenImages
* https://storage.googleapis.com/openimages/web/index.html