known_distance = 100 #CM
person_width = 35 #CM
bicycle_width = 1
car_width = 1
motorbike_width = 1
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

'''measured_distance = known_distance
   px_width = width detected by obj_detection'''

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

def distance_estimation(classes, boxes):
    for class in classes:
        if class
    
    
    print("Oh yeahh !")