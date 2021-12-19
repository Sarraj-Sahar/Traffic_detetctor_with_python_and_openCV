import cv2
from random import randrange

# our image
streetimg = cv2.imread('./images/cars4.jpg')

# video
vid = cv2.VideoCapture('./images/Crash recorded in TeslaCam.mp4')

# our pretrained classifiers
# car classifier
cars_classifier = cv2.CascadeClassifier('./cars.xml')

pedestrian_classifier = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
#
trained_face_data = cv2.CascadeClassifier(
    "../Face Detector/haarcascade_frontalface_default.xml")
#

trained_eyes_data = cv2.CascadeClassifier(
    "../Face Detector/haarcascade_eye.xml")


# make it grey

greyimg = cv2.cvtColor(streetimg, cv2.COLOR_BGR2GRAY)

# pedestrian coordinates
pedes_coordinates = pedestrian_classifier.detectMultiScale(
    streetimg)  # => an array of coordinates
#
face_coordinates = trained_face_data.detectMultiScale(
    streetimg)
#
eye_coordinates = trained_eyes_data.detectMultiScale(
    streetimg)

# cars coordinates
cars_coordinates = cars_classifier.detectMultiScale(
    greyimg)

print(pedes_coordinates)

# Framing
for (x, y, w, h) in pedes_coordinates:
    cv2.rectangle(streetimg, (x, y), (x+w, y+h), (0, 255, 255), 2)

#
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(streetimg, (x, y), (x+w, y+h), (0,
                  randrange(255), 0), 2)


#
for (x, y, w, h) in eye_coordinates:
    cv2.rectangle(streetimg, (x, y), (x+w, y+h), (0,
                  0, 255), 2)


for (x, y, w, h) in cars_coordinates:
    cv2.rectangle(greyimg, (x, y), (x+w, y+h), (255, 255, 58), 2)

# showing
# cv2.imshow("traffic", greyimg)
# cv2.waitKey()

####################################
####################################

# Live Version

while True:
    read_successfully, frame = vid.read()

    # grayScale
    if read_successfully:
        greyImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    live_cars_coordinates = cars_classifier.detectMultiScale(
        greyImage)
    for (x, y, w, h) in live_cars_coordinates:
        cv2.rectangle(greyImage, (x, y), (x+w, y+h), (148, 154, 154), 2)
    cv2.imshow('live video', greyImage)
    key = cv2.waitKey(1)

    # quit if q is pressed
    if key == 81 or key == 113:
        break
