from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import random

EVENTS_DB_URL = "sqlite:///./events.db"
engine = create_engine(EVENTS_DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer)
    event_type = Column(String)
    value = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

event_types = [
    "reliability", "spontaneity", "goal_orientation", "extraversion",
    "helpfulness", "achievement", "teamwork", "communication", "analytic", "stress_resilience"
]

user_profiles = {
    1: {"reliability": 5, "teamwork": 4, "achievement": 3},  # Mr. Zuverlässig
    2: {"teamwork": 1, "communication": 2, "analytic": 5},   # Einzelgänger
    3: {"achievement": 5, "goal_orientation": 5},            # Streber
    4: {"spontaneity": 5, "extraversion": 5},                # Partytyp
    5: {"helpfulness": 5, "communication": 4},               # Socializer
    6: {"analytic": 5, "reliability": 4},                    # Analyst
    7: {"stress_resilience": 1, "teamwork": 3},              # Sensibelchen
    8: {"goal_orientation": 1, "spontaneity": 5},            # Chaot
    9: {"reliability": 3, "helpfulness": 2},                 # Mitläufer
    10: {"achievement": 4, "teamwork": 5}                    # Teamleader
}

session = SessionLocal()
for user_id in range(1, 501):
    for _ in range(30):  # oder beliebig viele Events
        et = random.choice(event_types)
        # Check: Ist der User im user_profiles-Dict definiert?
        if user_id in user_profiles and et in user_profiles[user_id]:
            # 80% Wahrscheinlichkeit für Kern-Eigenschaften
            value = user_profiles[user_id][et] if random.random() < 0.8 else random.randint(1, 5)
        else:
            # Für alle User ohne spezielles Profil einfach random
            value = random.randint(1, 5)
        evt = Event(
            profile_id=user_id,
            event_type=et,
            value=value,
            timestamp=datetime.utcnow()
        )
        session.add(evt)
session.commit()
session.close()
print("Strukturierte Dummy-Events für 500 User erzeugt.")
