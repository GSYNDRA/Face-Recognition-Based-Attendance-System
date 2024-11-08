from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
import mysql.connector
from mysql.connector import Error

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
            if connection.is_connected():
                print("Success! Connected to MySQL database.")
                return connection
        except Error as e:
            print(f"fail to connect with MySQL database: {e}")
            return None
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():

    # selected_date = request.form.get('selected_date')
    # selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    # formatted_date = selected_date_obj.strftime('%Y-%m-%d')
    connection = connect_to_database()
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM Attendance WHERE journey_id = 16")
    attendance_data = cursor.fetchall()

    cursor.close()
    connection.close()

    if not attendance_data:
        return render_template('index.html', no_data=True)
    
    return render_template('index.html', attendance_data=attendance_data)

if __name__ == '__main__':
    app.run(debug=True)