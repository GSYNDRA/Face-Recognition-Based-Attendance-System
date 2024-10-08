import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import cvzone
import mysql.connector
from mysql.connector import Error
import random
import base64


# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        # GUI for background
        self.imgBackground = cv2.imread('resources/background.png')

        self.imgModeList = []
        for path in modePathList:
            self.imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

        self.current_student_id = 1
        self.last_student_id = None
        self.new_student_id = 0 

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.countdown = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 100000
    def connect_to_database(self):
        # """connect with database MySQL"""
        try:
            connection = mysql.connector.connect(
                host='127.0.0.1',  # address host of MySQL
                database='studentdb',  # name of database
                user='user',  # Username of MySQL
                password='user',  # Password of MySQL
                port='3309',
                raise_on_warnings=True  # keep informed if there's error 
            )
            # Database connection configuration
            if connection.is_connected():
                print("Success! Connected to MySQL database.")
                return connection
        except Error as e:
            print(f"fail to connect with MySQL database: {e}")
            return None
        
    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        connection = self.connect_to_database()
        if connection:
            cursor = connection.cursor()
            query = "SELECT feature_vector FROM Student ORDER BY student_id ASC LIMIT 2"
            # Execute the query
            cursor.execute(query)
            
            # Fetch all results from the query
            rows = cursor.fetchall()  # Fetch multiple rows
            for row in rows:
                feature_vector_str = row[0]  # Each row is a tuple, get the first element (feature_vector string)
                feature_vector_list = list(map(float, feature_vector_str.split(',')))  
                self.face_features_known_list.append(feature_vector_list)
            print(self.face_features_known_list)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            cursor.close()
            connection.close()
            return 1
        else:
            logging.warning("data is not found!")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # #  cv2 window / putText on cv2 window do not need
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (70, 200), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (70, 230), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (70, 260), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(img_rd, "Q: Quit", (70, 620), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
            
        
    # def get_ongoing_journey_id(self,connection, bus_id):
    #     cursor = connection.cursor()
    #     query = "SELECT journey_id FROM Journey WHERE status = 'ongoing' AND bus_id = %s"
    #     cursor.execute(query, (bus_id,))
    #     journey_id = cursor.fetchone()
    #     cursor.close()
    #     if journey_id:
    #         return int(journey_id[0])  
    #     else:
    #         return None  
        
    # def get_attendance_by_student_id(self, connection, student_id, journey_id):
    #     cursor = connection.cursor()
    #     query = "SELECT attendance_id, status FROM Attendance WHERE journey_id = %s AND student_id = %s"
    #     cursor.execute(query, (journey_id, student_id))
    #     attendance_id = cursor.fetchone()
    #     cursor.close()
    #     if attendance_id:
    #         return attendance_id  
    #     else:
    #         return None 
    
    # def get_info_by_student_id(self, student_id):
    #     connection = self.connect_to_database()
    #     if connection:
    #         cursor = connection.cursor()
    #         query = "SELECT name, avatar FROM Student WHERE student_id = %s"
    #         cursor.execute(query, (student_id,))
    #         infoStudent = cursor.fetchone()
    #         cursor.close()
    #         connection.close()
    #         if infoStudent:
    #             return infoStudent  
    #         else:
    #             return None 
        
    # def create_new_attendance(self, connection, student_id, journey_id, status, boarded, boarded_image):
    #     cursor = connection.cursor()
    #     query = """
    #     INSERT INTO Attendance (student_id, journey_id, status, boarded, boarded_image)
    #     VALUES (%s, %s, %s, %s, %s)
    #     """
    #     cursor.execute(query, (student_id, journey_id, status, boarded, boarded_image))
    #     connection.commit()
    #     cursor.close()

    # def update_attendance_status(self, connection, attendance_id, status, time, path):
    #     cursor = connection.cursor()
    #     query = """
    #     UPDATE Attendance
    #     SET status = %s, alighted = %s, alighted_image = %s
    #     WHERE attendance_id = %s
    #     """
    #     cursor.execute(query, (status, time, path, attendance_id))
    #     connection.commit()
    #     cursor.close()


    # def clear_alighted_info(self, connection, attendance_id, status, time, path):
    #     cursor = connection.cursor()
    #     query = """
    #     UPDATE Attendance
    #     SET status = %s, boarded= %s, boarded_image= %s, alighted = NULL, alighted_image = NULL
    #     WHERE attendance_id = %s
    #     """
    #     cursor.execute(query, (status, time, path, attendance_id))
    #     connection.commit()
    #     cursor.close()

    # # insert data in database
    # def attendance(self, time, id, path):
    #     connection = self.connect_to_database()
    #     if connection:
    #         journey_id = self.get_ongoing_journey_id(connection, 2)
    #         print(journey_id)
    #         if not journey_id:
    #             print("No active journey found.")
    #             return
    #         attendance_id = self.get_attendance_by_student_id(connection,id,journey_id)
    #         if not attendance_id:
    #             status = 'boarded'
    #             self.create_new_attendance(connection, id, journey_id, status, time, path)
    #         elif attendance_id[1] == 'boarded':
    #             status = 'alighted'
    #             self.update_attendance_status(connection, attendance_id[0], status, time, path)
    #         elif attendance_id[1] == 'alighted':
    #             status = 'boarded'
    #             self.clear_alighted_info(connection, attendance_id[0], status, time, path)
    #         connection.close()
    #     else:
    #         logging.warning("fail to connect")

    def get_ongoing_journey_id(self, connection, bus_id):
        # Truy vấn database để lấy journey ID
        cursor = connection.cursor()
        query = "SELECT journey_id FROM Journey WHERE status = 'ongoing' AND bus_id = %s"
        cursor.execute(query, (bus_id,))
        journey_id = cursor.fetchone()
        # Trả về journey ID dưới dạng int nếu tìm thấy, ngược lại trả về None
        # Kiểm tra nếu journey_id không phải là None trước khi truy cập phần tử [0]
        if journey_id:
            return int(journey_id[0])  # Trả về journey ID dưới dạng int nếu có giá trị
        else:
            return None  # Ngược lại trả về None
        
    def get_attendance_by_student_id(self, connection, student_id, journey_id):
        cursor = connection.cursor()
        query = "SELECT attendance_id, status FROM Attendance WHERE journey_id = %s AND student_id = %s"
        cursor.execute(query, (journey_id, student_id))
        attendance_id = cursor.fetchone()
        # Trả về journey ID dưới dạng int nếu tìm thấy, ngược lại trả về None
        # Kiểm tra nếu journey_id không phải là None trước khi truy cập phần tử [0]
        if attendance_id:
            print(attendance_id)
            return attendance_id  # Trả về journey ID dưới dạng int nếu có giá trị
        else:
            return None  # Ngược lại trả về None
        
    def create_new_attendance(self, connection, student_id, student_name, journey_id, status, boarded, image):
        cursor = connection.cursor()
        insert_attendance_query = """
        INSERT INTO Attendance (student_id, journey_id, status, boarded)
        VALUES (%s, %s, %s, %s)
        """

        cursor.execute(insert_attendance_query, (student_id, journey_id, status, boarded))
        attendance_id = cursor.lastrowid
        insert_noti_query = """

        INSERT INTO Notification (attendance_id, time_stamp, message, image, status)
        VALUES (%s, %s, %s, %s, 'common')
        """
        cursor.execute(insert_noti_query, (attendance_id, boarded, f"{student_name} has boarded the bus", image))
        connection.commit()

    def update_attendance_status(self, connection, student_name, attendance_id, status, time, image):
        cursor = connection.cursor()
        update_attendance_query = """
        UPDATE Attendance
        SET status = %s, alighted = %s
        WHERE attendance_id = %s
        """
        cursor.execute(update_attendance_query, (status, time, attendance_id))
        connection.commit()
        
        insert_noti_query = """
        INSERT INTO Notification (attendance_id, time_stamp, message, image, status)
        VALUES (%s, %s, %s, %s, 'common')
        """
        cursor.execute(insert_noti_query, (attendance_id, time, f"{student_name} has alighted the bus", image))
        connection.commit()


    def clear_alighted_info(self, connection, student_name, attendance_id, status, time, image):
        cursor = connection.cursor()
        query = """
        UPDATE Attendance
        SET status = %s, boarded= %s, alighted = NULL
        WHERE attendance_id = %s
        """
        cursor.execute(query, (status, time, attendance_id))

        insert_noti_query = """
        INSERT INTO Notification (attendance_id, time_stamp, message, image, status)
        VALUES (%s, %s, %s, %s, 'common')
        """
        cursor.execute(insert_noti_query, (attendance_id, time, f"{student_name} has boarded the bus", image))

        connection.commit()

    # insert data in database
    def attendance(self, time, id, path):
        connection = self.connect_to_database()
        if connection:
            journey_id = self.get_ongoing_journey_id(connection, 2)
            print(journey_id)
            if not journey_id:
                print("No active journey found.")
                return
            attendance_id = self.get_attendance_by_student_id(connection,id,journey_id)
            name = self.get_name_by_student_id(id)
            print(name)
            # print(attendance_id[0], attendance_id[1])
            if not attendance_id:
                status = 'boarded'
                self.create_new_attendance(connection, id, name, journey_id, status, time, path)
            elif attendance_id[1] == 'boarded':
                status = 'alighted'
                self.update_attendance_status(connection, name, attendance_id[0], status, time, path)
            elif attendance_id[1] == 'alighted':
                status = 'boarded'
                self.clear_alighted_info(connection, name, attendance_id[0], status, time, path)

        else:
            logging.warning("fail to connect")

    def get_name_by_student_id(self, student_id):
        connection = self.connect_to_database()
        if connection:
            cursor = connection.cursor()
            query = "SELECT name FROM Student WHERE student_id = %s"
            cursor.execute(query, (student_id,))
            name_student = cursor.fetchone()
            cursor.close()
            connection.close()
            if name_student:
                return name_student[0]
            else:
                return None 


    def save_recognized_face(self, img, person_name):
        # Get the current date
        current_date = datetime.datetime.now().strftime("%Y_%m_%d")

        # Create a directory for the person if it doesn't exist
        person_folder = f"data/recognized_faces_check_in/{person_name}"
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Create the path for the image file with the name as {person_name}_realtime_checkin_{current_date}.png
        file_path = f"{person_folder}/{person_name}_realtime_checkin_{current_date}.png"

        # Check if the image for the current date already exists
        if not os.path.exists(file_path):
            # Save the frame
            cv2.imwrite(file_path, img)
            print(f"Saved {person_name}'s face for {current_date}")
        else:
            print(f"{person_name}'s face for {current_date} has already been saved.")

        # Return the file path (whether it was saved or not)
        return file_path
    #  Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1.  Get faces known from "features.all.csv"
        if self.get_face_database():
            self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]
            while stream.isOpened():
                self.frame_cnt += 1
                self.reclassify_interval_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                # flipped_frame = cv2.flip(img_rd, 1)

                self.imgBackground[162:162+480,55:55+640] = img_rd
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)
                if(len(faces) > 0):
                    bbox = (55+faces[0].left(), 162+faces[0].top(), faces[0].right() - faces[0].left(), faces[0].bottom() - faces[0].top())
                    img_rd = cvzone.cornerRect(img_rd, bbox, rt=0, colorR=(255, 255, 255))
                    if (55 <= bbox[0] <= 55 + 640 - bbox[2] and  # Kiểm tra cạnh trái và phải (x trong phạm vi và không vượt quá cạnh phải)
                    162 <= bbox[1] <= 162 + 480 - bbox[3] and # Kiểm tra cạnh trên và dưới (y trong phạm vi và không vượt quá cạnh dưới)
                    bbox[2] <= 640 and                        # Chiều rộng của bbox không vượt quá chiều rộng của camera
                    bbox[3] <= 480):                          # Chiều cao của bbox không vượt quá chiều cao của camera
                        cvzone.cornerRect(self.imgBackground, bbox, rt=0, colorR=(255, 255, 255))

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []
                

                # 6.1  self.current_student_id == self.last_student_id
                if (self.current_student_id == self.last_student_id) and self.reclassify_interval_cnt <= 100:
                    if self.reclassify_interval_cnt == 1:
                        print("getting data of student")
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]
                    elif self.reclassify_interval_cnt <= 20:
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]
                        print("putText for background with cv2")
                    elif 20 < self.reclassify_interval_cnt <= 60:
                        infoStudent = self.get_info_by_student_id(self.current_student_id)
                        avatar = base64.b64decode(infoStudent[1])
                        nparr = np.frombuffer(avatar, np.uint8)
                        imgStudent = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        imgStudent = cv2.resize(imgStudent, (216, 216))
                        
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[1]
                        cv2.putText(self.imgBackground, str(infoStudent[0]), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(self.imgBackground, str(self.current_student_id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        self.imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                        print("showing the info frame of student")
                    elif 60 < self.reclassify_interval_cnt <= 80:
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[2]
                        print("showing checked image")
                    else:
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]
                    self.draw_note(self.imgBackground)

                elif self.reclassify_interval_cnt > 100 and self.countdown >= 0:
                    if 15 < self.countdown <= 50:
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[3]
                    else:
                       self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]
                    self.countdown -= 1
                    self.draw_note(self.imgBackground)

                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt += 1
                    
                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        # for i in range(len(faces)):
                        shape = predictor(img_rd, faces[0])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape))
                        self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        # for k in range(len(faces)):
                        logging.debug("  For face %d in current frame:",1)
                        self.current_frame_face_centroid_list.append(
                            [int(faces[0].left() + faces[0].right()) / 2,
                             int(faces[0].top() + faces[0].bottom()) / 2])

                        self.current_frame_face_X_e_distance_list = []

                        # 6.2.2.2  Positions of faces captured
                        self.current_frame_face_position_list.append(tuple(
                            [faces[0].left(), int(faces[0].bottom() + (faces[0].bottom() - faces[0].top()) / 4)]))

                        # 6.2.2.3 
                        # For every faces detected, compare the faces in the database
                        for i in range(len(self.face_features_known_list)):
                            # 
                            if str(self.face_features_known_list[i]) != '0.0':
                                e_distance_tmp = self.return_euclidean_distance(
                                self.current_frame_face_feature_list[0],
                                self.face_features_known_list[i])
                                logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                            else:
                                #  person_X
                                self.current_frame_face_X_e_distance_list.append(999999999)
                        # 6.2.2.4 / Find the one with minimum e distance
                        self.new_student_id  = self.current_frame_face_X_e_distance_list.index(
                            min(self.current_frame_face_X_e_distance_list)) + 1
                        print(min(self.current_frame_face_X_e_distance_list), self.new_student_id)
                        self.current_student_id = self.new_student_id
                        # nếu số càng bé thì dễ miss những người có trong danh sách số càng lớn dễ lọc ra những người không được đăng kí
                        if min(self.current_frame_face_X_e_distance_list) >= 0.3:
                            self.new_student_id = -1
                        print(self.current_student_id, self.new_student_id)
                        if min(self.current_frame_face_X_e_distance_list) < 0.3 and (self.current_student_id == self.last_student_id):
                            self.countdown = 50
                        # print(self.current_student_id, self.last_student_id, new_student_id)
                        # chỉnh xuống 0.3 or 0.35 nếu muốn tắng độ chính xác
                        elif min(self.current_frame_face_X_e_distance_list) < 0.3 and (self.current_student_id != self.last_student_id):
                            # update the id of student
                            self.reclassify_interval_cnt = 1
                            current_time = datetime.datetime.now()
                            self.last_student_id = self.current_student_id
                            print(self.current_student_id, self.last_student_id)

                            saved_image_path = self.save_recognized_face(img_rd, f"{self.current_student_id}_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}")

                            self.attendance(current_time.strftime("%Y-%m-%d %H:%M:%S"), self.current_student_id, saved_image_path)

                    # nếu không có dòng này thì ko update hình mà hình bình thương luôn trả về 0
                    if (self.new_student_id != self.current_student_id) and self.reclassify_interval_cnt > 100:
                        print("showing activate image and UNKNOWN PERSON")
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]
                        # 7.  / Add note on cv2 window
                    self.draw_note(self.imgBackground)
                    
                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", self.imgBackground)

                logging.debug("Frame ends\n\n")

    


    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process(cap)
        cap.set(3, 640)
        cap.set(4, 490)

        cap.release()
        cv2.destroyAllWindows()
    
   


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()