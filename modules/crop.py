import numpy as np
import cv2
from skimage import io
from skimage.transform import resize

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
    

def crop_face(img, shape):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_frontal.detectMultiScale(gray)    
    face = rect(faces[0])
    ymin = face[0][0]
    ymax = face[1][0]
    xmin = face[0][1]
    xmax = face[3][1]
    dx = 0.05 * (xmax - xmin)
    dy = 0.05 * (ymax - ymin)
    xmin -= dx
    xmax += dx
    ymin -= dy
    xmax += dy
    ratio = shape[0] / shape[1]
    ratio1 = (ymax - ymin) / (xmax - xmin)
    if ratio1 < ratio:
        y_needed = (xmax - xmin) * ratio
        dy = 0.5 * (y_needed - (ymax - ymin))
        ymax += dy
        ymin -= dy
    elif (ratio1 > ratio):
        x_needed = (ymax - ymin) / ratio
        dx = 0.5 * (x_needed - (xmax - xmin))
        xmax += dx
        xmin -= dx
    print(xmin, xmax, ymin, ymax)
    return img[int(ymin):int(ymax), int(xmin):int(xmax)]