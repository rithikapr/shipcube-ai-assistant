import sqlite3

conn = sqlite3.connect('data/shipcube.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(chats)")
columns = cursor.fetchall()

print("Columns in chats table")
for col in columns:
    print(col[1])

conn.close()