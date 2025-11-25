import sqlite3
from werkzeug.security import generate_password_hash

DB = "data/shipcube.db"
username = "testuser"
password = "Test@1234"

conn = sqlite3.connect(DB)
cur = conn.cursor()
pw_hash = generate_password_hash(password)
cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pw_hash))
conn.commit()
conn.close()
print("Created user:", username)
