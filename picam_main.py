# Dependences:
#
# $ sudo apt update
# $ sudo apt install build-essentials
# $ sudo apt install libatlas-base-dev
# $ sudo apt install python3-pip
# $ pip3 install tflite-runtime
# $ pip3 install opencv-python==4.4.0.46
# $ pip3 install pillow
# $ pip3 install numpy

import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time
#from PIL import Image
#from PIL import ImageFont, ImageDraw

from picamera2 import Picamera2, Preview, MappedArray

#normalSize = (640, 480)
#imW, imH = 640, 480
lowresSize = (640, 480)

#rectangles = []


def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

#model = ("/home/pi/rms/models/ssd_mobilenet_v3/model.tflite")
model = "/home/pi/rms/mobilenet_v2.tflite"
labels = ReadLabelFile("/home/pi/rms/coco_labels.txt")

def detect(img,model,labels):
    interpreter = tflite.Interpreter(model, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    picture = cv2.resize(img, (width, height))
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True


    initial_w, initial_h = lowresSize

    input_data = np.expand_dims(picture, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    #print(input_details, input_data)
    
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    rectangles = []
    for i in range(int(num_boxes)):
        top, left, bottom, right = detected_boxes[0][i]
        #print(detected_boxes[0][i])
        classId = int(detected_classes[0][i])
        score = detected_scores[0][i]
        if score > 0.4:
            xmin = left * initial_w
            ymin = bottom * initial_h
            xmax = right * initial_w
            ymax = top * initial_h
            box = [xmin, ymin, xmax, ymax]
            rectangles.append(box)
            cv2.rectangle(img,rectangles[i], (10, 255, 0), 2)
            #print(int(score[i]*100))
            cv2.rectangle(img,[i], (10, 255, 0), 2)
            label = '%s: %d%%' % (labels[classId], int(score*100))
            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)
            label_ymin = max(int(ymin), labelSize[1]+10)
            #print(rectangles)    
            cv2.putText(img, label, (int(xmin), label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 225), 2)            
            print(labels[classId], 'score = ', score)
            #rectangles[-1].append(labels[classId])
            return img

def main():

    picam2 = Picamera2()
    config = picam2.preview_configuration(main={"size": (320,240), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    frame_count=0
    #cap = cv2.VideoCapture("/home/pi/rms/videos/15fps.mp4")
    starting_time = time.time()
    #img = cv2.imread("1.png")
    while True:
        frame_count+=1
        
        if frame_count%10==0:
            #ret, img = cap.read()    
            img = picam2.capture_array()
            #grey = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,lowresSize)
            result = detect(img,model,labels)

            cv2.imshow("frame",img)
            endingTime = time.time() - starting_time
            fps = frame_count/endingTime 
            print(fps)
    #cv2.imwrite("output.jpg", img)
            #print(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    #cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
