import dlib
import numpy as np
import cv2
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import requests
from io import BytesIO

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

#  Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

#  Get face landmarks
predictor = dlib.shape_predictor("data_dlib/shape_predictor_68_face_landmarks.dat")

#  Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def return_128d_features_from_url(url):
    # Download the image from the URL
    response = requests.get(url)
    
    if response.status_code == 200:
        img_data = BytesIO(response.content)
        img_rd = np.array(Image.open(img_data))  # Open the image as an array
        
        # Convert from RGB (PIL) to BGR (OpenCV)
        img_rd = cv2.cvtColor(img_rd, cv2.COLOR_RGB2BGR)
        
        # Detect faces in the image
        faces = detector(img_rd, 1)

        logging.info("%-40s %-20s", " Image with faces detected:", url)

        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
            face_descriptor = np.array(face_descriptor) 
            feature_vector_str = ','.join(map(str, face_descriptor))  # string of feature vector

        else:
            feature_vector_str = np.zeros(128, dtype=object, order='C')
            logging.warning("no face detected")

        return feature_vector_str
    else:
        logging.error(f"Failed to download the image from {url}")
        return None
# def return_128d_features_from_url(url, features_list=None):
#     if features_list is None:
#         features_list = []  # Initialize an empty list if not passed
    
#     # Download the image from the URL
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         img_data = BytesIO(response.content)
#         img_rd = np.array(Image.open(img_data))  # Open the image as an array
        
#         # Convert from RGB (PIL) to BGR (OpenCV)
#         img_rd = cv2.cvtColor(img_rd, cv2.COLOR_RGB2BGR)
        
#         # Detect faces in the image
#         faces = detector(img_rd, 1)

#         logging.info("%-40s %-20s", " Image with faces detected:", url)

#         # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
#         if len(faces) != 0:
#             shape = predictor(img_rd, faces[0])
#             face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
#             face_descriptor = np.array(face_descriptor)  # Convert to numpy array
            
#             # Append the face descriptor to the features list
#             features_list.append(face_descriptor)
            
#             # Compute the mean of the collected feature vectors
#             features_mean = np.array(features_list, dtype=object).mean(axis=0)

#             # Convert the mean feature vector to a string for return
#             feature_vector_str = ','.join(map(str, features_mean))  # String of mean feature vector
#             logging.info(f"Mean feature vector for {url}: {feature_vector_str}")
            
#         else:
#             feature_vector_str = np.zeros(128, dtype=object, order='C')
#             logging.warning("No face detected")

#         return feature_vector_str
#     else:
#         logging.error(f"Failed to download the image from {url}")
#         return None

url = "https://studenttestdb.blob.core.windows.net/avatars/STUDENT_TRACKING-avatar-parentId=1-studentId=15-1732012022799-img_face_1.jpg"
features = return_128d_features_from_url(url)
if features is not None:
    print("Face features:", features)
else:
    print("No face detected or failed to process the image.")