import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time
from distance_estimation import distance_estimation
from OCR import OCR

dist_estimate = ''
normalSize = (640, 480)
#lowresSize = (320, 240)

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

model = ""
labels =  ReadLabelFile()

def draw_boxes(img,num_boxes,detected_boxes,detected_classes,detected_scores):
        cv2.rectangle(img,box, (10, 255, 0), 2)
        label = '%s: %d%%' % (labels[classId], int(score*100))
            
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)
        label_ymin = max(int(ymin), labelSize[1]+10)
        cv2.putText(img, label, (int(xmin), label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 225), 2)
        


def detect(img,model,labels):
    interpreter = tflite.Interpreter(model, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(output_details[1])
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    picture = cv2.resize(img, (width, height))
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True


    initial_w, initial_h = normalSize

    input_data = np.expand_dims(picture, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    rectangles = []
    classes ={}
    for i in range(int(num_boxes)):
        bottom,left, top, right = detected_boxes[0][i]
        classId = int(detected_classes[0][i])
        score = detected_scores[0][i]
        if score > 0.4:
            xmin = left * initial_w
            ymin = bottom * initial_h
            xmax = right * initial_w
            ymax = top * initial_h
            box = [xmin, ymin, xmax, ymax]
            rectangles.append(box)
            #class_names.append(labels[classId])
            #classIds.append(classId)
            classes[classId] = labels[classId]

    return img,rectangles,classes

def main():
    cap = cv2.VideoCapture(0)
    starting_time = time.time()
    while True:
        ret, img = cap.read()
        img = cv2.resize(img,normalSize)
        result,boxes,classes = detect(img,model,labels)
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

