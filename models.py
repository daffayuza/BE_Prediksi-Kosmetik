from sqlalchemy import Column, Integer, Float, JSON
from database import Base
from datetime import datetime
from sqlalchemy.types import DateTime


class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True, index=True)
    pengunjung = Column(Integer)
    tayangan = Column(Integer)
    pesanan = Column(Integer)
    terjual = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelStore(Base):
    __tablename__ = "model_store"
    id = Column(Integer, primary_key=True, index=True)
    # coefficients = Column(JSON)
    intercept = Column(Float)
    b1 = Column(Float)
    b2 = Column(Float)
    b3 = Column(Float)
    r2_score = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelEvaluation(Base):
    __tablename__ = "model_evaluation"

    id = Column(Integer, primary_key=True, index=True)
    r2_score = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
