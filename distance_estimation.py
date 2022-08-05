from object_det import detect
import cv2
import numpy as np
known_distance = 100 #CM
person_width = 35 #CM
mobile_width = 12 #CM
#bicycle_width = 1
#car_width = 1
#motorbike_width = 1
bottle_width = 1
backpack_width = 1
pottedplant_width = 1
chair_width = 1
tvmonitor_width = 1
laptop_width = 1
mouse_width = 1
remote_width = 1
keyboard_width = 1
cellphone_width = 1
book_width  = 1

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv2.FONT_HERSHEY_COMPLEX

'''measured_distance = known_distance
   width_in_rf = width detected by obj_detection'''

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    #print(distance)
    return distance


def distance_estimation(img, data):
    print(data)
    ref_mobile = cv2.imread("E:\\YOLO\\Final_Code\\ReferenceImages\\back_crop.jpg")
    ref_person = cv2.imread("E:\YOLO\Final_Code\ReferenceImages\image14.png")
    mobile_data = detect(ref_mobile)[3]
    #print(mobile_data)
    mobile_width_in_rf = mobile_data[0][1]

    person_data = detect(ref_person)[3]
    person_width_in_rf = person_data[0][1]

    print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")
    
    # finding focal length 
    focal_person = focal_length_finder(known_distance, person_width, person_width_in_rf)
    focal_mobile = focal_length_finder(known_distance, mobile_width, mobile_width_in_rf)

    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, person_width, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, mobile_width, d[1])
            x, y = d[2]
        cv2.rectangle(img, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv2.putText(img, f'Dis: {distance} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
            
    
    
    #print("Oh yeahh !")