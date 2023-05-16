import sqlite3
import os

def create_database():
    # Check if the database file already exists
    if os.path.exists('music_database.db'):
        print("Database file already exists.")
        return

    # Create a connection to the database
    conn = sqlite3.connect('music_database.db')

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create the table for storing music details
    cursor.execute('''CREATE TABLE music (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        music_name TEXT,
                        composer_name TEXT,
                        audio BLOB
                    )''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Database created successfully.")

def add_music(music_name, composer_name, audio_file_path):
    # Check if the database file exists
    if not os.path.exists('music_database.db'):
        print("Database file does not exist. Create the database first.")
        return

    # Read the audio file data
    with open(audio_file_path, 'rb') as file:
        audio_data = file.read()

    # Establish a connection to the database
    conn = sqlite3.connect('music_database.db')

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Insert the music details into the database
    cursor.execute("INSERT INTO music (music_name, composer_name, audio) VALUES (?, ?, ?)",
                   (music_name, composer_name, audio_data))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Music added successfully.")

# Create the database
create_database()

# Add a music entry to the database
music_name = "Pick Your Poison"
composer_name = "Black Pistol Fire"
audio_file_path = "C:\\Users\\Administrateur\\Downloads\\Projet_IA\\Audios\\pick_your_poison_black_pistol_fire.m4a"
add_music(music_name, composer_name, audio_file_path)
