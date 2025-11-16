from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, String
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import pytz

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(pytz.UTC))

    # Relasi
    training_data = relationship("TrainingData", back_populates="product")
    testing_data = relationship("TestingData", back_populates="product")
    model_store = relationship("ModelStore", back_populates="product")
    prediction_history = relationship("PredictionHistory", back_populates="product")

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    pengunjung = Column(Integer)
    tayangan = Column(Integer)
    pesanan = Column(Integer)
    terjual = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relasi balik
    product = relationship("Product", back_populates="training_data")

class TestingData(Base):
    __tablename__ = "testing_data"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    pengunjung = Column(Integer)
    tayangan = Column(Integer)
    pesanan = Column(Integer)
    terjual = Column(Integer)
    predicted = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="testing_data")

class ModelStore(Base):
    __tablename__ = "model_store"
    id = Column(Integer, primary_key=True, index=True)
    intercept = Column(Float)
    b1 = Column(Float)
    b2 = Column(Float)
    b3 = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(pytz.UTC))

    product = relationship("Product", back_populates="model_store")
    evaluation = relationship("ModelEvaluation", back_populates="model")

class ModelEvaluation(Base):
    __tablename__ = "model_evaluation"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model_store.id"))
    r2_score = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelStore", back_populates="evaluation")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)