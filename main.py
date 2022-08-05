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
    frame_count = 0
    cap = cv2.VideoCapture(0)
    starting_time = time.time()
    
    while True:
        frame_count += 1
        ret, img = cap.read()
        img = cv2.resize(img,normalSize)
        if (frame_count % 1 == 0):
            result,boxes,class_names, data_list = detect(img)
            #draw_boxes(img,num_boxes,detected_boxes,detected_classes,detected_scores)
            if (dist_estimate == '1'):
                distance_estimation(result,data_list)
            if ('sign board' in class_names):
                OCR(img)
            cv2.imshow("frame",img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ending_time = time.time() - starting_time
        fps = frame_count/ending_time
    print(fps)            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

