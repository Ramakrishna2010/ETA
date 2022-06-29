from PIL import Image
import pytesseract
import cv2
import numpy as np
import pyttsx3

engine = pyttsx3.init()

pytesseract.pytesseract.tesseract_cmd = "E://YOLO//rms//tesseract_OCR//tesseract.exe"

#filename = "E://YOLO//OCR//mining.jpeg"
#img = np.array(Image.open(filename))
def OCR(img):
    text_img = np.array(img)
    gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh)
    #print(text)

    engine.say(text)
    engine.runAndWait()

#cv2.imshow('img',thresh)
#cv2.waitKey(0)