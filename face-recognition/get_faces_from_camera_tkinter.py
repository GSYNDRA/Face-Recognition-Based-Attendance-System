import dlib
import numpy as np
import cv2
import csv
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import subprocess
import mysql.connector
from mysql.connector import Error

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

#  Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

#  Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

#  Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def get_latest_folder(path):
        # Lấy danh sách tất cả folder trong thư mục
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        
        if folders:
            # Tìm folder có thời gian sửa đổi mới nhất
            latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(path, f)))
            return latest_folder
        else:
            return None
        
class Face_Register:
    def __init__(self):

        self.current_frame_faces_cnt = 0  #  cnt for counting faces in current frame    
        self.existing_faces_cnt = 0  # cnt for counting saved faces
        self.ss_cnt = 0  #  cnt for screen shots

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register")

        # PLease modify window size here if needed
        self.win.geometry("1200x600")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.input_id = tk.Entry(self.frame_right_info)
        self.input_id_number = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""

        self.font = cv2.FONT_ITALIC

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)  # Get video stream from camera

        # self.cap = cv2.VideoCapture("test.mp4")   # Input local video
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
             # Database connection configuration
            if connection.is_connected():
                print("Success! Connected to MySQL database.")
                return connection
        except Error as e:
            print(f"fail to connect with MySQL database: {e}")
            return None

    def check_student_exists(self, name, student_id):
        # "Check the student exist in student table or not"
        connection = self.connect_to_database()
        if connection:
            cursor = connection.cursor()
            query = "SELECT * FROM Student WHERE name = %s AND student_id = %s"
            cursor.execute(query, (name, student_id))
            result = cursor.fetchone()
            cursor.close()
            connection.close()

            if result:
                print("receive")
                return True
            else:
                print("reject")
                return False
        return False

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        #  Get ID from input => check if it is number or not ?
        try:
            self.input_id_number = int(self.input_id.get())  # convert into int
        except ValueError:
            self.log_all["text"] = "ID must be a number!"  # Display the error if ID is not a number 
            return
        #Check the student exist in database or not ?
        student_exists = self.check_student_exists(self.input_name_char, self.input_id_number)

        if (student_exists == True):
            self.create_face_folder()  # if existed => create folder
            self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)
        else:
            #Display the notification if the student doesn't exist
            self.label_warning['text'] = f"Student {self.input_name_char} with ID {self.input_id_number} doesn't exist!"
            self.label_warning['fg'] = 'red'
 
    # def extract_feature(self):
    #     logging.basicConfig(level=logging.INFO)
    #     #  Get the order of latest person
    #     person_list = os.listdir("data/data_faces_from_camera/")
    #     print(person_list[-1])
    #     person_list.sort()

    #     with open("data/features_all.csv", "w", newline="") as csvfile:
    #         writer = csv.writer(csvfile)
    #         for person in person_list:
    #             # Get the mean/average features of face/personX, it will be a list with a length of 128D
    #             logging.info("%sperson_%s", path_images_from_camera, person)
    #             features_mean_personX = self.return_features_mean_personX(path_images_from_camera + person)

    #             if len(person.split('_', 2)) == 2:
    #                 # "person_x"
    #                 person_name = person
    #             else:
    #                 # "person_x_tom"
    #                 person_name = person.split('_', 2)[-1]
    #             features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
    #             # features_mean_personX will be 129D, person name + 128 features
    #             writer.writerow(features_mean_personX)
    #             logging.info('\n')
    #         logging.info("Save all the features of faces registered into: data/features_all.csv")

    def update_feature_vector(self, student_id, feature_vector):
        connection = self.connect_to_database()
        if connection:
            try:
                cursor = connection.cursor()

                # convert numpy array (or list) into string or BLOB to save
                feature_vector_str = ','.join(map(str, feature_vector))  # string of feature vector
                
                # query SQL to update feature_vector
                query = "UPDATE Student SET feature_vector = %s WHERE student_id = %s"
                
                #execute SQL query
                cursor.execute(query, (feature_vector_str, student_id))
                connection.commit()  # commit change
                
                print(f"Updated feature vector for student ID: {student_id}")
            except Error as e:
                print(f"Failed to update feature vector: {e}")
            finally:
                cursor.close()
                connection.close()
        else:
            print("Connection to the database failed.")

    
    
    def extract_feature(self):
        logging.basicConfig(level=logging.INFO)
        # person_list = os.listdir("data/data_faces_from_camera/")
        latest_folder = get_latest_folder(path_images_from_camera)
                
        # get mean feature vector of identified person 
        features_mean_personX = self.return_features_mean_personX(path_images_from_camera + latest_folder)

        # now, save features_mean_personX into database
        self.update_feature_vector(self.input_id_number, features_mean_personX)


    #  Return 128D features for single image
    def return_128d_features(self, path_img):
        img_rd = cv2.imread(path_img)
        faces = detector(img_rd, 1)

        logging.info("%-40s %-20s", " Image with faces detected:", path_img)

        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            logging.warning("no face")
        return face_descriptor


    #   Return the mean value of 128D face descriptor for person 
    def return_features_mean_personX(self, path_face_personX):
        features_list_personX = []
        photos_list = os.listdir(path_face_personX)
        if photos_list:
            for i in range(len(photos_list)):
                #  return_128d_features()  128D  / Get 128D features for single image of personX
                logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + photos_list[i])
                features_128d = self.return_128d_features(path_face_personX + "/" + photos_list[i])
                #  Jump if no face detected from image
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            logging.warning(" Warning: No images in%s/", path_face_personX)

    
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
            print(features_mean_personX)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        return features_mean_personX

    #  Delete old face folders
    def GUI_clear_data(self):
        #  "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        # tk.Button(self.frame_right_info,
        #           text='Clear',
        #           command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name and create folders for face
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input name").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        # Input for ID (added beside Name input)
        tk.Label(self.frame_right_info, text="ID: ").grid(row=8, column=2, sticky=tk.W, padx=5, pady=0)
        self.input_id.grid(row=8, column=3, sticky=tk.W, padx=0, pady=2)

        # tk.Button(self.frame_right_info,
        #           text='Input',
        #           command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)
        # Button to submit Name and ID
        tk.Button(self.frame_right_info,text='Input',command=self.GUI_get_input_name).grid(row=8, column=4, padx=5)
        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 3: Save face image").grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='Save current face',
                  command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W)
        # Button to extract features (added beside Save current face)
        tk.Button(self.frame_right_info,
              text='Extract features',
              command=self.extract_feature).grid(row=10, column=1, padx=5, pady=2)
        # Show log in GUI
        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.existing_faces_cnt = max(person_num_list)

        # Start from person_1
        else:
            self.existing_faces_cnt = 0

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        #  Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        #  Create the folders for saving faces
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt) + "_" + \
                                    self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  #  Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created


    def save_current_face(self):
        if not self.face_folder_created_flag:
            self.log_all["text"] = "Please run step 2!"
            return

        if self.current_frame_faces_cnt != 1:
            self.log_all["text"] = "No face in current frame!"
            return

        if self.out_of_range_flag:
            self.log_all["text"] = "Please do not out of range!"
            return

        self.ss_cnt += 1

        self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3), np.uint8)
        
        for ii in range(self.face_ROI_height * 2):
            for jj in range(self.face_ROI_width * 2):
                self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                    self.face_ROI_width_start - self.ww + jj]
        
        # Cập nhật log để hiển thị đường dẫn của ảnh đã lưu
        self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg\"" + " saved!"
        
        # Chuyển đổi hình ảnh từ BGR sang RGB
        self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)
        
        # Lưu hình ảnh vào thư mục
        cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
        
        # Ghi lại thông tin lưu ảnh vào log
        logging.info("%-40s %s/img_face_%s.jpg", "Save into：", str(self.current_face_dir), str(self.ss_cnt) + ".jpg")


    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640,480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    #  Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # Get frame
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            #  Face detected
            if len(faces) != 0:
                #   Show the ROI of faces
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    #  Compute the size of rectangle box
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # If the size of ROI > 480x640
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            # Convert PIL.Image.Image to PIL.Image.PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
