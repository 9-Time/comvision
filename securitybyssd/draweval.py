import os
import cv2
import pandas as pd

EVAL_RESULTS = 'evalresults/det_test_Beer.txt'
IMAGE_DIRECTORY = 'data/open_images/test'
OUTPUT_DIRECTORY = 'evaloutput'
THRESHOLD = 0.5

if not os.path.exists(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)
all_txt = os.listdir(EVAL_RESULTS)
df = pd.read_csv(EVAL_RESULTS, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
df['x1'] = df['x1'].astype(int)
df['y1'] = df['y1'].astype(int)
df['x2'] = df['x2'].astype(int)
df['y2'] = df['y2'].astype(int)

for image_id, g in df.groupby('ImageID'):
    image = cv2.imread(os.path.join(IMAGE_DIRECTORY, image_id + ".jpg"))
    for row in g.itertuples():
        if row.Prob < THRESHOLD:
            continue
        cv2.rectangle(image, (row.x1, row.y1), (row.x2, row.y2), (255, 255, 0), 4)
        label = f"{row.Prob:.2f}"
        cv2.putText(image, label,
                    (row.x1 + 20, row.y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, image_id + ".jpg"), image)
print(f"Task Done. Processed {df.shape[0]} bounding boxes.")