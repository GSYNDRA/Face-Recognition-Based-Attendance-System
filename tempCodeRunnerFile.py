


# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

#  Path of cropped faces
path_images_from_camera = os.path.join(pwd, "data/data_faces_from_camera")

#  Get face landmarks
predictor = dlib.shape_predictor(pwd + '/data/data_dlib/shape_predictor_68_face_landmarks.dat')

#  Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(pwd + "/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")