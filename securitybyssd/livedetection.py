from pkg import vgg_ssd
import cv2

MODEL = 'checkpoint/vgg-Epoch-10-Loss-3.0180892944335938.pth'

cap = cv2.VideoCapture(0)   # capture from camera 0 (change if you have multiple cams)
cap.set(3, 1920)
cap.set(4, 1080)

class_names = [name.strip() for name in open('checkpoint/open-images-model-labels.txt').readlines()]
num_classes = len(class_names)

net = vgg_ssd.create_vgg_ssd(len(class_names), is_test=True)
net.load(MODEL)
predictor = vgg_ssd.create_vgg_ssd_predictor(net, candidate_size=200)

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