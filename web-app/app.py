from flask import Flask, render_template, request
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            database='studentdb',
            user='user',
            password='user',
            port='3309',
            raise_on_warnings=True
        )
        if connection.is_connected():
            print("Success! Connected to MySQL database.")
            return connection
    except Error as e:
        print(f"Failed to connect to MySQL database: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    journeyID = request.form.get('journeyID')
    connection = connect_to_database()
    
    if connection:
        cursor = connection.cursor()
        query = "SELECT * FROM Attendance WHERE journey_id = %s"
        cursor.execute(query, (journeyID,))
        attendance_data = cursor.fetchall()
        cursor.close()
        connection.close()

        if not attendance_data:
            return render_template('index.html', no_data=True)
        return render_template('index.html', attendance_data=attendance_data)

    return render_template('index.html', no_data=True)

if __name__ == '__main__':
    app.run(debug=True)
