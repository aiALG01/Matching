from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Optional
from datetime import datetime
import numpy as np
from numpy.linalg import norm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random

# ----------- Profile DB Setup (test.db) -----------
DATABASE_URL = "sqlite:///./test.db"
engine_profiles = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocalProfiles = sessionmaker(bind=engine_profiles)
BaseProfiles = declarative_base()

# ----------- Events DB Setup (events.db) ----------
EVENTS_DB_URL = "sqlite:///./events.db"
engine_events = create_engine(EVENTS_DB_URL, connect_args={"check_same_thread": False})
SessionLocalEvents = sessionmaker(bind=engine_events)
BaseEvents = declarative_base()

# ----------- Profile Model -----------
class Profile(BaseProfiles):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    # Selbsteinschätzung
    reliability = Column(Integer)
    spontaneity = Column(Integer)
    goal_orientation = Column(Integer)
    extroversion = Column(Integer)
    helpfulness = Column(Integer)
    achievement = Column(Integer)
    teamwork = Column(Integer)
    communication = Column(Integer)
    analytic = Column(Integer)
    stress_resilience = Column(Integer)
    # Fremdeinschätzung
    reliability_verified = Column(Integer, nullable=True)
    spontaneity_verified = Column(Integer, nullable=True)
    goal_orientation_verified = Column(Integer, nullable=True)
    extroversion_verified = Column(Integer, nullable=True)
    helpfulness_verified = Column(Integer, nullable=True)
    achievement_verified = Column(Integer, nullable=True)
    teamwork_verified = Column(Integer, nullable=True)
    communication_verified = Column(Integer, nullable=True)
    analytic_verified = Column(Integer, nullable=True)
    stress_resilience_verified = Column(Integer, nullable=True)
    # Status: 0=nur Selbst, 1=verifiziert
    reliability_status = Column(Integer, default=0)
    spontaneity_status = Column(Integer, default=0)
    goal_orientation_status = Column(Integer, default=0)
    extroversion_status = Column(Integer, default=0)
    helpfulness_status = Column(Integer, default=0)
    achievement_status = Column(Integer, default=0)
    teamwork_status = Column(Integer, default=0)
    communication_status = Column(Integer, default=0)
    analytic_status = Column(Integer, default=0)
    stress_resilience_status = Column(Integer, default=0)
    # SVM-Label als Gruppenlabel
    svm_label = Column(String, nullable=True)

# ----------- Events Model -----------
class Event(BaseEvents):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer)
    event_type = Column(String)
    value = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ----------- Tabellen erzeugen -----------
BaseProfiles.metadata.create_all(bind=engine_profiles)
BaseEvents.metadata.create_all(bind=engine_events)

# ----------- SVM Dummy-Setup -----------
LABELS = ["harmonisch", "dominant", "initiativ", "analytisch", "ausgleichend"]
def create_dummy_dataset(n=100):
    X = np.random.randint(1, 6, (n, 10))
    y = np.random.choice(LABELS, n)
    return X, y

X_dummy, y_dummy = create_dummy_dataset(200)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)
svm = SVC(kernel="linear", probability=True)
svm.fit(X_scaled, y_dummy)

def predict_label(params):
    x = np.array([params]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    label = svm.predict(x_scaled)[0]
    return label

# ----------- FastAPI Setup -----------
app = FastAPI(title="Matching-API")

# ----------- Schemas -----------
class ProfileCreate(BaseModel):
    username: str
    reliability: int
    spontaneity: int
    goal_orientation: int
    extroversion: int
    helpfulness: int
    achievement: int
    teamwork: int
    communication: int
    analytic: int
    stress_resilience: int

class ProfileOut(ProfileCreate):
    id: int
    svm_label: Optional[str]

# ----------- Endpoints -----------

# Profil anlegen
@app.post("/profile/create", response_model=ProfileOut)
def create_profile(profile: ProfileCreate):
    db = SessionLocalProfiles()
    db_profile = db.query(Profile).filter(Profile.username == profile.username).first()
    if db_profile:
        db.close()
        raise HTTPException(status_code=409, detail="Username already exists")
    params = [
        profile.reliability, profile.spontaneity, profile.goal_orientation,
        profile.extroversion, profile.helpfulness, profile.achievement,
        profile.teamwork, profile.communication, profile.analytic, profile.stress_resilience
    ]
    svm_label = predict_label(params)
    new_profile = Profile(
        username=profile.username,
        reliability=profile.reliability,
        spontaneity=profile.spontaneity,
        goal_orientation=profile.goal_orientation,
        extroversion=profile.extroversion,
        helpfulness=profile.helpfulness,
        achievement=profile.achievement,
        teamwork=profile.teamwork,
        communication=profile.communication,
        analytic=profile.analytic,
        stress_resilience=profile.stress_resilience,
        svm_label=svm_label
    )
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    out = ProfileOut(
        id=new_profile.id,
        username=new_profile.username,
        reliability=new_profile.reliability,
        spontaneity=new_profile.spontaneity,
        goal_orientation=new_profile.goal_orientation,
        extroversion=new_profile.extroversion,
        helpfulness=new_profile.helpfulness,
        achievement=new_profile.achievement,
        teamwork=new_profile.teamwork,
        communication=new_profile.communication,
        analytic=new_profile.analytic,
        stress_resilience=new_profile.stress_resilience,
        svm_label=new_profile.svm_label
    )
    db.close()
    return out

# Alle Profile anzeigen
@app.get("/profiles")
def list_profiles():
    db = SessionLocalProfiles()
    profiles = db.query(Profile).all()
    db.close()
    result = []
    for p in profiles:
        result.append({
            "id": p.id,
            "username": p.username,
            "svm_label": p.svm_label,
            "reliability": p.reliability,
            "spontaneity": p.spontaneity,
            "goal_orientation": p.goal_orientation,
            "extroversion": p.extroversion,
            "helpfulness": p.helpfulness,
            "achievement": p.achievement,
            "teamwork": p.teamwork,
            "communication": p.communication,
            "analytic": p.analytic,
            "stress_resilience": p.stress_resilience
        })
    return result

# Komplette Profilansicht inkl. Fremdeinschätzung
@app.get("/profile/{profile_id}/full")
def get_full_profile(profile_id: int):
    db = SessionLocalProfiles()
    p = db.query(Profile).filter(Profile.id == profile_id).first()
    db.close()
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    def status_str(s):
        return "Verifiziert" if s == 1 else "Nur Selbsteinschätzung"
    return {
        "username": p.username,
        "persönlichkeitsprofil": [
            {
                "eigenschaft": "Zuverlässigkeit",
                "selbst": p.reliability,
                "fremd": p.reliability_verified,
                "status": status_str(p.reliability_status)
            },
            {
                "eigenschaft": "Spontanität",
                "selbst": p.spontaneity,
                "fremd": p.spontaneity_verified,
                "status": status_str(p.spontaneity_status)
            },
            {
                "eigenschaft": "Lernzielstrebigkeit",
                "selbst": p.goal_orientation,
                "fremd": p.goal_orientation_verified,
                "status": status_str(p.goal_orientation_status)
            },
            {
                "eigenschaft": "Extrovertiertheit",
                "selbst": p.extroversion,
                "fremd": p.extroversion_verified,
                "status": status_str(p.extroversion_status)
            },
            {
                "eigenschaft": "Hilfsbereitschaft",
                "selbst": p.helpfulness,
                "fremd": p.helpfulness_verified,
                "status": status_str(p.helpfulness_status)
            },
            {
                "eigenschaft": "Leistungswunsch",
                "selbst": p.achievement,
                "fremd": p.achievement_verified,
                "status": status_str(p.achievement_status)
            },
            {
                "eigenschaft": "Teamfähigkeit",
                "selbst": p.teamwork,
                "fremd": p.teamwork_verified,
                "status": status_str(p.teamwork_status)
            },
            {
                "eigenschaft": "Kommunikation",
                "selbst": p.communication,
                "fremd": p.communication_verified,
                "status": status_str(p.communication_status)
            },
            {
                "eigenschaft": "Analytik",
                "selbst": p.analytic,
                "fremd": p.analytic_verified,
                "status": status_str(p.analytic_status)
            },
            {
                "eigenschaft": "Stressresistenz",
                "selbst": p.stress_resilience,
                "fremd": p.stress_resilience_verified,
                "status": status_str(p.stress_resilience_status)
            },
        ],
        "label": p.svm_label
    }

# Dummy-Events erzeugen
@app.post("/event/dummy/{profile_id}")
def create_dummy_events(profile_id: int, n: int = 20):
    event_types = [
        "reliability", "spontaneity", "goal_orientation", "extroversion",
        "helpfulness", "achievement", "teamwork", "communication", "analytic", "stress_resilience"
    ]
    db_events = SessionLocalEvents()
    for _ in range(n):
        et = random.choice(event_types)
        val = random.randint(1, 5)
        evt = Event(profile_id=profile_id, event_type=et, value=val)
        db_events.add(evt)
    db_events.commit()
    db_events.close()
    return {"status": f"{n} dummy events created for profile {profile_id}"}

# Events eines Profils abrufen
@app.get("/event/{profile_id}")
def get_events(profile_id: int):
    db_events = SessionLocalEvents()
    events = db_events.query(Event).filter(Event.profile_id == profile_id).all()
    db_events.close()
    return [
        {"event_type": e.event_type, "value": e.value, "timestamp": e.timestamp} for e in events
    ]

# RL-basierte Fremdeinschätzung setzen (Mittelwert je Eigenschaft)
@app.post("/profile/{profile_id}/rl_verify")
def rl_verify(profile_id: int):
    # 1. User holen
    db_profiles = SessionLocalProfiles()
    profile = db_profiles.query(Profile).filter(Profile.id == profile_id).first()
    db_profiles.close()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # 2. Events holen
    db_events = SessionLocalEvents()
    events = db_events.query(Event).filter(Event.profile_id == profile_id).all()
    db_events.close()

    import numpy as np
    event_types = [
        "reliability", "spontaneity", "goal_orientation", "extroversion",
        "helpfulness", "achievement", "teamwork", "communication", "analytic", "stress_resilience"
    ]
    vals = {et: [] for et in event_types}
    for e in events:
        vals[e.event_type].append(e.value)
    means = {et: int(round(np.mean(vals[et]))) if vals[et] else None for et in event_types}

    # 3. Fremdeinschätzung updaten
    db_profiles = SessionLocalProfiles()
    p = db_profiles.query(Profile).filter(Profile.id == profile_id).first()
    for et in event_types:
        setattr(p, f"{et}_verified", means[et])
        setattr(p, f"{et}_status", 1 if means[et] is not None else 0)
    db_profiles.commit()
    db_profiles.close()
    return {"status": "RL-Fremdeinschätzung gesetzt", "werte": means}

# Matching Score (Cosine Similarity)
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

@app.get("/match_score")
def match_score(profile_id_1: int, profile_id_2: int):
    db = SessionLocalProfiles()
    p1 = db.query(Profile).filter(Profile.id == profile_id_1).first()
    p2 = db.query(Profile).filter(Profile.id == profile_id_2).first()
    db.close()
    if not p1 or not p2:
        raise HTTPException(status_code=404, detail="Profile not found")
    vec1 = [
        p1.reliability, p1.spontaneity, p1.goal_orientation, p1.extroversion,
        p1.helpfulness, p1.achievement, p1.teamwork, p1.communication,
        p1.analytic, p1.stress_resilience
    ]
    vec2 = [
        p2.reliability, p2.spontaneity, p2.goal_orientation, p2.extroversion,
        p2.helpfulness, p2.achievement, p2.teamwork, p2.communication,
        p2.analytic, p2.stress_resilience
    ]
    score = cosine_similarity(vec1, vec2)
    return {
        "profile_id_1": profile_id_1,
        "profile_id_2": profile_id_2,
        "score": round(score, 3)
    }
