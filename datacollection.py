#now we are going to start with the webcam
import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector #for detecting the hand signs

cap = cv2.VideoCapture(0)


detector = HandDetector(maxHands=1) #as we are using only 1 hand for the detection purpose
offset=20 #it makes crop image look better
imgsize = 300
counter =0
#to save images
folder = 'data/c'
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # for finding the hand in the image

    # now we will crop our image

    if hands:
        hand = hands[0]  # as we have only 1 hand
        # now we will get the bounding box information
        x,y,w,h = hand['bbox']  # here x and y are positions and w and h are width and heigth and we will ask dictionary to give us values
        # now we will crop the image

        # we will create an image by ourselves to counter the height and width or rectangle shaped images
        # we will create a white image and will give it dimesnions using numpy matrix

        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255  #our image is an square aaray of 400*400
        # np.uint8=numpy unsigned integer of 8 bits as colors rangr from 0 to 255
        imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset] # starting height is y and ending is y+h
        # starting width is x and ending is x+w
        # this will give us bounding box that we require

        #now we will cropped image on top of white image

        imagecropshape=imgcrop.shape

        # imgwhite[0:imagecropshape[0],0:imagecropshape[1]]=imgcrop #put it inside if statement

        # here imagecrop[0] gives the height of image and imagecrop[1] width of the image

        # now we will check height and width if height is bigger than width we will make it 300 if the width
        # is bigger than the height we will stretch the width to 300

        aspectratio = h/w # height divided by width if value greater than 1 means height is more

        if aspectratio > 1:
            k = imgsize/h  # k is constant
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imgsize))
            imgresizeshape = imgresize.shape
            # no we will center our image
            wgap = math.ceil((imgsize-wcal)/2)  # as image and center will always  have some gap

            # imgwhite[0:imgresizeshape[0], 0:imgresizeshape[1]] = imgresize now modifying it

            # as height will always be 300 and width starting poistion will be gap and ending will be gap+wcal

            imgwhite[: ,wgap:wgap+wcal]= imgresize

        #now for width

        else:
            k = imgsize / w  # k is constant
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize,hcal))
            imgresizeshape = imgresize.shape
            # no we will center our image
            hgap = math.ceil((imgsize - hcal) / 2)  # as image and center will always  have some gap

            # imgwhite[0:imgresizeshape[0], 0:imgresizeshape[1]] = imgresize now modifying it

            # as height will always be 300 and width starting poistion will be gap and ending will be gap+wcal

            imgwhite[hgap:hgap+hcal, : ] = imgresize

        cv2.imshow("imgcrop",imgcrop)
        cv2.imshow("imgwhite",imgwhite)


    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('s') : # if we press s then it will save the imgwhite in a directory
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgwhite)
        #time.time will give us unique value that does not repeat
        print(counter)






