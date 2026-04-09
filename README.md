# Face Recognition-Based School Bus Attendance System

## Problem Statement
Incidents of students being forgotten on school buses can happen when monitoring is manual and inconsistent. Traditional attendance methods are prone to human error, especially during busy pickup/drop-off periods. This creates a serious safety risk, particularly for younger students who may not be able to alert adults quickly.

## Solution Overview
This project provides an AI-assisted attendance workflow for school bus operations:
- Registers student faces and extracts biometric feature vectors.
- Performs real-time face recognition from a bus camera feed.
- Automatically updates boarding/alighting status in the database.
- Creates notification records with timestamped messages and captured evidence images.

The goal is to reduce missed check-ins and improve student safety with automated, auditable tracking.

## Main Features
- Face registration UI (`Tkinter`) for collecting student face images.
- 128D facial feature extraction using `dlib` models.
- Real-time recognition with `OpenCV + dlib`.
- Automatic attendance state transitions (`boarded` <-> `alighted`).
- Notification logging with image path and event messages.
- SQL schema and seed data included for quick setup.

## Tech Stack
- Python (`OpenCV`, `dlib`, `numpy`, `pandas`, `Tkinter`)
- MySQL
- SQL schema/data script: `face-recognition/studentdb(new).sql`

## Project Structure
```text
AI PROJECT/
|- face-recognition/
|  |- attendance_taker.py                 # Real-time recognition + attendance updates
|  |- get_faces_from_camera_tkinter.py    # Face enrollment and feature extraction
|  |- studentdb(new).sql                  # Database schema + seed data
|  |- requirements.txt
|  |- data_dlib/                          # dlib landmark + face recognition models
|  |- data/                               # Face dataset and recognized face captures
|  `- resources/                          # Background and UI mode assets
|- web-app/
|  `- templates/index.html                # Attendance table template (frontend)
`- README.md
```

## How It Works
1. **Enroll Student Face Data**
   - Run `get_faces_from_camera_tkinter.py`.
   - Input student name + ID (must exist in `Student` table).
   - Capture multiple face images.
   - Extract mean feature vector and update `Student.feature_vector`.

2. **Run Real-Time Attendance**
   - Run `attendance_taker.py`.
   - Camera stream detects and recognizes student faces.
   - System checks ongoing journey and updates `Attendance` + `Notification` tables.
   - Recognized face snapshots are saved under:
     - `face-recognition/data/recognized_faces_check_in/`

3. **Review Attendance Data**
   - Attendance and notification logs are stored in MySQL.
   - The template in `web-app/templates/index.html` can be integrated with your backend to display journey attendance.

## Setup Instructions
### 1. Prerequisites
- Python 3.8+ (Windows recommended for current scripts)
- MySQL Server
- Webcam

### 2. Install Python Dependencies
```bash
cd face-recognition
pip install -r requirements.txt
```

If you get missing-module errors, install additional runtime packages used in scripts:
```bash
pip install mysql-connector-python pillow cvzone requests
```

### 3. Prepare dlib Model Files
Make sure these files exist:
- `face-recognition/data_dlib/shape_predictor_68_face_landmarks.dat`
- `face-recognition/data_dlib/dlib_face_recognition_resnet_model_v1.dat`

### 4. Create Database and Import SQL
Create a database named `studentdb`, then import:
```bash
mysql -u user -p -P 3309 studentdb < "studentdb(new).sql"
```

If your MySQL credentials/port differ, update the connection settings in:
- `face-recognition/get_faces_from_camera_tkinter.py`
- `face-recognition/attendance_taker.py`

Current defaults in code:
- Host: `127.0.0.1`
- Port: `3309`
- User: `user`
- Password: `user`
- Database: `studentdb`

## Run the System
From the `face-recognition` folder:
```bash
python get_faces_from_camera_tkinter.py
python attendance_taker.py
```

## Database Notes
Core tables used by this workflow:
- `Student` (includes `avatar` and `feature_vector`)
- `Journey` (must have an `ongoing` record)
- `Attendance`
- `Notification`

## Current Assumptions and Limitations
- Camera source is currently local webcam (`cv2.VideoCapture(0)`).
- Recognition logic is tuned for one primary face at a time in frame.
- Attendance script currently queries ongoing journey for a fixed bus (`bus_id = 2`) unless you modify the code.
- Accuracy depends on lighting quality, camera angle, and enrollment image quality.

## Safety Impact
By automatically logging boarding/alighting events and producing time-stamped notifications, this system helps reduce the risk of students being left unattended on school buses and supports faster intervention when anomalies occur.


