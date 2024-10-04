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
import base64
import datetime



def connect_to_database():
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

def get_face_database():
        connection = connect_to_database()
        if connection:
            cursor = connection.cursor()
            query = "SELECT feature_vector FROM Student ORDER BY student_id ASC LIMIT 3"
            # Execute the query
            cursor.execute(query)
            
            # Fetch all results from the query
            rows = cursor.fetchall()  # Fetch multiple rows

            feature_vectors = []
            for row in rows:
                feature_vector_str = row[0]  # Each row is a tuple, get the first element (feature_vector string)
                feature_vector_list = list(map(float, feature_vector_str.split(',')))  
                feature_vectors.append(feature_vector_list)
            print(feature_vectors)
            cursor.close()
            connection.close()
        else:
            print("Connection to the database failed.")

# get_face_database()
# feature_vector_str = "-0.07291254810988904,0.09814447909593582,0.04164188727736473,-0.03032720759510994,-0.08739526718854904,-0.021730219572782518,-0.08023011237382889,-0.07676087841391563,0.15864129811525346,-0.07720943912863731,0.2101309686899185,-0.045178354158997534,-0.19419622123241426,-0.05581501051783562,0.0023039035499095918,0.15076130628585815,-0.15448689609766006,-0.07483271211385727,-0.05351696349680424,-0.017872093245387077,0.04898047596216202,0.10757856369018555,0.006610976904630661,0.04755304753780365,-0.08291495144367218,-0.3256256878376007,-0.08896077573299407,-0.08421722948551177,0.03262838274240494,-0.0517760306596756,-0.08885978758335114,0.0398830208927393,-0.10954421013593674,-0.07400212064385414,0.07577913627028465,0.10132819637656212,-0.044513695128262046,-0.06869197338819504,0.20825932919979095,-0.01777622401714325,-0.23031625151634216,0.05762802213430405,0.07085249125957489,0.23452866673469544,0.20699126422405242,0.044849943928420545,0.008524622209370137,-0.1650708496570587,0.1652251422405243,-0.17353440225124359,0.038934686407446864,0.16917292177677154,0.040782394260168074,0.07740098536014557,-0.0009966363199055196,-0.1418252944946289,-0.0006085699424147606,0.13934493958950042,-0.15052019953727722,0.003933273255825043,0.06883103922009468,-0.10945117324590684,-0.0070304999127984045,-0.04945396557450295,0.17987096011638642,0.044139917194843295,-0.12014059126377105,-0.20718172788619996,0.054522180929780006,-0.18691153228282928,-0.10081201791763306,0.12008628100156785,-0.14469468593597412,-0.18128948509693146,-0.31031373143196106,0.03985025435686111,0.4085328638553619,0.079571932554245,-0.15876261591911317,0.07368333712220192,-0.03518896773457527,-0.07707180008292198,0.12210916429758072,0.18181610405445098,0.0034797679632902145,-0.03384139239788055,-0.09035638272762299,-0.005653287470340729,0.26818978786468506,-0.06978061720728874,0.005180848762392998,0.2125589966773987,0.024470752477645873,0.12244168668985367,0.0039020229130983354,0.08194630295038223,-0.08179232031106949,0.014461276307702064,-0.10251933932304383,-0.02177266925573349,0.05738770132884383,-0.027369354292750357,-0.0016224164515733718,0.11503860652446747,-0.20010510683059693,0.17171115875244142,0.008001527190208435,0.01888180784881115,0.05460267383605242,-0.05055541731417179,-0.04429072178900242,-0.04759119153022766,0.1780476838350296,-0.21727508902549744,0.1721254765987396,0.15059131383895874,0.08745820820331573,0.14473328590393067,0.10346627160906792,0.083634664863348,-0.015022392012178898,-0.02081657648086548,-0.22616093456745148,-0.01897650696337223,0.07748851478099823,0.006020051706582308,0.07278410941362382,0.0054262328892946245"
# # # Bước 1: Tách chuỗi thành danh sách các số thực
# feature_vector_list = list(map(float, feature_vector_str.split(',')))

# # Bước 2: Chuyển danh sách thành numpy array để so sánh
# feature_vector_array = np.array(feature_vector_list)
# Đường dẫn tới hình ảnh
# image_path = "data/recognized_faces_check_in/1_2024_10_04_16_17_00/1_2024_10_04_16_17_00_realtime_checkin_2024_10_04.png"
# image = cv2.imread(image_path)

# # Hiển thị ảnh
# cv2.imshow("Image", image)
# cv2.waitKey(0)  # Đợi cho đến khi nhấn phím để đóng cửa sổ
# cv2.destroyAllWindows()
# # Đọc file ảnh và mã hóa thành Base64
# with open(image_path, "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# In ra chuỗi Base64
# print(encoded_string)
# In ra kết quả
# print(feature_vector_array)
base64_string = "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"
# Bước 1: Loại bỏ tiền tố không cần thiết
# base64_data = base64_string.split(",")[1]
# Bước 2: Giải mã chuỗi Base64 thành dữ liệu nhị phân
image_data = base64.b64decode(base64_string)

# Bước 3: Chuyển dữ liệu nhị phân thành mảng NumPy
nparr = np.frombuffer(image_data, np.uint8)

# Bước 4: Giải mã mảng NumPy thành hình ảnh OpenCV
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Bước 5: Hiển thị hình ảnh sử dụng OpenCV
cv2.imshow("Decoded Image", img)

# Đợi cho tới khi bạn bấm phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)

# Đóng tất cả các cửa sổ hiển thị của OpenCV
cv2.destroyAllWindows()
# current_time = datetime.datetime.now()
# current_time1 = current_time.strftime("%Y-%m-%d %H:%M:%S")

# print(current_time1)

def get_ongoing_journey_id(connection, bus_id):
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
def get_attendance_by_student_id(connection, student_id, journey_id):
        cursor = connection.cursor()
        query = "SELECT attendance_id, status FROM Attendance WHERE journey_id = %s AND student_id = %s"
        cursor.execute(query, (journey_id, student_id))
        attendance_id = cursor.fetchone()
        # Trả về journey ID dưới dạng int nếu tìm thấy, ngược lại trả về None
        # Kiểm tra nếu journey_id không phải là None trước khi truy cập phần tử [0]
        if attendance_id:
            return attendance_id  # Trả về journey ID dưới dạng int nếu có giá trị
        else:
            return None  # Ngược lại trả về None
def create_new_attendance(connection, student_id, journey_id, status, boarded, boarded_image):
        cursor = connection.cursor()
        query = """
        INSERT INTO Attendance (student_id, journey_id, status, boarded, boarded_image)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (student_id, journey_id, status, boarded, boarded_image))
        connection.commit()
def update_attendance_status(connection, attendance_id, status, time, path):
        cursor = connection.cursor()
        query = """
        UPDATE Attendance
        SET status = %s, alighted = %s, alighted_image = %s
        WHERE attendance_id = %s
        """
        cursor.execute(query, (status, time, path, attendance_id))
        connection.commit()


def clear_alighted_info(connection, attendance_id, status, time, path):
        cursor = connection.cursor()
        query = """
        UPDATE Attendance
        SET status = %s, boarded= %s, boarded_image= %s, alighted = NULL, alighted_image = NULL
        WHERE attendance_id = %s
        """
        cursor.execute(query, (status, time, path, attendance_id))
        connection.commit()

    # insert data in database
def attendance(time, id, path):
        connection = connect_to_database()
        if connection:
            journey_id = get_ongoing_journey_id(connection, 2)
            print(journey_id)
            if not journey_id:
                print("No active journey found.")
                return
            attendance_id = get_attendance_by_student_id(connection,id,journey_id)
            print(attendance_id[0], attendance_id[1])
            if not attendance_id:
                status = 'boarded'
                create_new_attendance(connection, id, journey_id, status, time, path)
            elif attendance_id[1] == 'boarded':
                status = 'alighted'
                update_attendance_status(connection, attendance_id[0], status, time, path)
            elif attendance_id[1] == 'alighted':
                status = 'boarded'
                clear_alighted_info(connection, attendance_id[0], status, time, path)

        else:
            logging.warning("fail to connect")
def get_info_by_student_id(student_id):
        connection = connect_to_database()
        if connection:
            cursor = connection.cursor()
            query = "SELECT student_id, name, avatar FROM Student WHERE student_id = %s"
            cursor.execute(query, (student_id,))
            infoStudent = cursor.fetchone()
            cursor.close()
            connection.close()
            if infoStudent:
                return infoStudent  
            else:
                return None 
infoStudent = get_info_by_student_id(1)  
print(infoStudent)
# current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# image_path = "data/recognized_faces_check_in/1_2024_10_04_16_17_00/1_2024_10_04_16_17_00_realtime_checkin_2024_10_04.png"

# attendance(current_time, 1, image_path)

