import time
from glob import iglob

import cv2
from matplotlib import pyplot as plt


def main():

    for path in iglob("query_images/*"):
        img = cv2.imread(path)
        detector = cv2.AKAZE_create()
        kp = detector.detect(img)
        img_with_key = cv2.drawKeypoints(img, kp, None)
        img_with_key = cv2.cvtColor(img_with_key, cv2.COLOR_BGR2RGB)
        plt.imshow(img_with_key)
        plt.pause(5)
        print(f"num kps: {len(kp)}")


if __name__ == "__main__":
    main()
