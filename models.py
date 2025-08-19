from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from database import Base
from datetime import datetime
import pytz

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True, index=True)
    pengunjung = Column(Integer)
    tayangan = Column(Integer)
    pesanan = Column(Integer)
    terjual = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class TestingData(Base):
    __tablename__ = "testing_data"
    id = Column(Integer, primary_key=True, index=True)
    pengunjung = Column(Integer)
    tayangan = Column(Integer)
    pesanan = Column(Integer)
    terjual = Column(Integer)
    predicted = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelStore(Base):
    __tablename__ = "model_store"
    id = Column(Integer, primary_key=True, index=True)
    intercept = Column(Float)
    b1 = Column(Float)
    b2 = Column(Float)
    b3 = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(pytz.UTC))

class ModelEvaluation(Base):
    __tablename__ = "model_evaluation"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model_store.id"))
    r2_score = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    mape = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)