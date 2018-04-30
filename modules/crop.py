import numpy as np
import cv2
from skimage import io

"""
This code uses open_cv face_frontal detector to extract faces from images.
NOTE:
In case it fails to detect a face in a photo it may return strange parts of the photo.
Better check that the result is reasonable.
"""

face_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def rect(face):
    return np.array([
        [face[1], face[0]],
        [face[1] + face[3], face[0]],
        [face[1], face[0] + face[2]],
        [face[1] + face[3], face[0] + face[2]]
    ])
    

def crop_face(img):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_frontal.detectMultiScale(gray)    
    face = rect(faces[0])
    return img[face[0][0]:face[1][0], face[0][1]:face[3][1]]