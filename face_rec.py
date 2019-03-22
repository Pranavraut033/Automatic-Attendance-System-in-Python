#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle

import cv2
import face_recognition
from imutils import paths
from collections import Counter
from IPython.display import clear_output


FACE_CASCADE = "classifier/haarcascade_frontalface_default.xml"
IMG_SIZE = 128

TRAINING_DATASET = 'dataset_training'
TESTING_DATASET = 'dataset_testing'
MODEL_NAME = 'model_encodings.pickle'

FONT = cv2.FONT_HERSHEY_SIMPLEX
FACE_CLASSIFIER = cv2.CascadeClassifier(FACE_CASCADE)


# In[ ]:


# grab the paths to the input images in our dataset
print("[INFO] fetching images...")
image_paths = list(paths.list_images(TRAINING_DATASET))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
num_images = len(image_paths)
for (i, image_path) in enumerate(image_paths):
    # extract the person name from the image path
    name = image_path.split(os.path.sep)[-2]
    print("[INFO] processing image (%s) %d/%d..." %
          (name, i + 1, num_images), end="\r")

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(image, model='cnn')

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(image, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("\n[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(MODEL_NAME, "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] DONE.")


# In[ ]:


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(MODEL_NAME, "rb").read())

print("[INFO] recognizing faces...")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH , 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT  , 720)
# cam.set(3, 1280)

# cam.set(4, 1024)

cv2.namedWindow("recognize faces")

num_names = Counter(data['names'])

while True:
    ret, org = cam.read()
    if not ret:
        break
    k = cv2.waitKey(1)
    
    cv2.imshow("recognize faces", org)

    if k%256 == 27:
        print("\nEscape hit, closing...")
        break

    # load the input image and convert it from BGR to RGB

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face 
    faces = FACE_CLASSIFIER.detectMultiScale(org)
    clear_output(wait=True)

    for (i, f) in enumerate(faces):
        x, y, w, h = [ v for v in f ]

        sub_face = org[y : y+h+15, x : x + w + 15]
#         cv2.imshow(str(i), sub_face)

        image = cv2.resize(sub_face, (IMG_SIZE,IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(image, model='hop')
        encodings = face_recognition.face_encodings(image, boxes)

        names = []
        boxes = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            num_prop = 0
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
                num_prop = counts[name] / num_names[name]
                # update the list of names

                print("[INFO] detected %s with '%f' accuracy ..." % (name, num_prop))

            if(num_prop > 0.85):
                print("[INFO] Found %s (%f)." % (name, num_prop))

            boxes.append((y, x + w, y + h, x))
            names.append(name if num_prop > 0.8 else "Unknown")

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(org, (left, top), (right, bottom), (255, 255, 255), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(org, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            # show the output image
            cv2.imshow("recognize faces", org)

cam.release()

cv2.destroyAllWindows()


# In[ ]:




