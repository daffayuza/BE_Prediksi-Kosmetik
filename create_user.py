from database import SessionLocal, engine, Base
from models import User
from werkzeug.security import generate_password_hash

# Pastikan tabel dibuat
Base.metadata.create_all(bind=engine)

# Buat session
db = SessionLocal()

# Hash password sebelum disimpan
hashed_pw = generate_password_hash("admin123")

new_user = User(
    username="admin",
    password_hash=hashed_pw
)

db.add(new_user)
db.commit()
db.close()

print("User admin berhasil dibuat.")
