from pkg.vggssd import *
from pkg.predictor import *
import cv2
import torch
from pkg.config import vgg_ssd_config as config

MODEL = 'checkpoint/v2-Epoch-39-Loss-7.86013218334743.pth'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

cap = cv2.VideoCapture(0)  
cap.set(3, 1920)
cap.set(4, 1080)

class_names = [name.strip() for name in open('checkpoint/open-images-model-labels.txt').readlines()]
num_classes = len(class_names)

net = VGGSSD(len(class_names), device, config=config, is_test=True)
net.load(MODEL)
predictor = Predictor(net, device, config.image_size, config.image_mean, nms_method='hard')

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    print('Detect Objects: {:d}.'.format(labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()