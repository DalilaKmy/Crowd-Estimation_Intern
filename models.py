from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime

class Flight(Base):
    __tablename__ = 'names'

    id = Column(Integer, primary_key=True)
    status = Column(String(50))
    peopleIn = Column(Integer)
    peopleOut = Column(Integer)
    totalPeople = Column(Integer)
    dateTime = Column(DateTime)

    def __init__(self, status=None, peopleIn=None, peopleOut=None, totalPeople=None, dateTime=None):
        self.status = status
        self.peopleIn = peopleIn
        self.peopleOut = peopleOut
        self.totalPeople = totalPeople
        self.dateTime = dateTime

    def __repr__(self):
        return '<Flight %r>' % (self.status)