#import tflite_runtime.interpreter as tflite
#import numpy as np
import cv2
import time
from object_det import *
from distance_estimation import distance_estimation
from OCR import OCR

dist_estimate = ''
normalSize = (640, 480)
#lowresSize = (320, 240)
    
def main():
    cap = cv2.VideoCapture(0)
    starting_time = time.time()
    while True:
        ret, img = cap.read()
        img = cv2.resize(img,normalSize)
        result,boxes,class_names, data_list = detect(img)
        #draw_boxes(img,num_boxes,detected_boxes,detected_classes,detected_scores)
        if (dist_estimate == '1'):
            distance_estimation(boxes)
        if ('sign board' in class_names):
            OCR(img)
        cv2.imshow("frame",img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

