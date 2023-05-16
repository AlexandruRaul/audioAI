import sqlite3

# Open a connection to the SQLite database
conn = sqlite3.connect('audio.db')

# Retrieve the audio file data
audio_file_path = 'path_to_your_audio_file.mp3'
with open(audio_file_path, 'rb') as file:
    audio_file_data = file.read()

# Prepare the SQL INSERT statement
insert_query = "INSERT INTO YourTable (audio_data) VALUES (?)"

# Execute the INSERT statement
conn.execute(insert_query, (audio_file_data,))

# Commit the transaction
conn.commit()

# Close the database connection
conn.close()
