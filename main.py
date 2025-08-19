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
from sqlalchemy import ForeignKey
from typing import List
from fastapi import Query

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

# Gruppen-Datenbank
GROUPS_DB_URL = "sqlite:///./groups.db"
engine_groups = create_engine(GROUPS_DB_URL, connect_args={"check_same_thread": False})

# Separate Basisklasse für die Gruppen-DB
BaseGroups = declarative_base()

# ----------- Profile Model -----------
class Profile(BaseProfiles):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    university = Column(String)
    course_of_study = Column(String)

    # Selbsteinschätzung
    extraversion = Column(Integer)
    agreeableness = Column(Integer)
    conscientiousness = Column(Integer)
    emotional_stability = Column(Integer)
    openness = Column(Integer)
    team_orientation = Column(Integer)
    motivation = Column(Integer)
    communication = Column(Integer)
    cohesion = Column(Integer)
    goal_orientation = Column(Integer)

    # Fremdeinschätzung
    extraversion_verified = Column(Integer, nullable=True)
    agreeableness_verified = Column(Integer, nullable=True)
    conscientiousness_verified = Column(Integer, nullable=True)
    emotional_stability_verified = Column(Integer, nullable=True)
    openness_verified = Column(Integer, nullable=True)
    team_orientation_verified = Column(Integer, nullable=True)
    motivation_verified = Column(Integer, nullable=True)
    communication_verified = Column(Integer, nullable=True)
    cohesion_verified = Column(Integer, nullable=True)
    goal_orientation_verified = Column(Integer, nullable=True)

    # Status: 0=nur Selbst, 1=verifiziert
    extraversion_status = Column(Integer, default=0)
    agreeableness_status = Column(Integer, default=0)
    conscientiousness_status = Column(Integer, default=0)
    emotional_stability_status = Column(Integer, default=0)
    openness_status = Column(Integer, default=0)
    team_orientation_status = Column(Integer, default=0)
    motivation_status = Column(Integer, default=0)
    communication_status = Column(Integer, default=0)
    cohesion_status = Column(Integer, default=0)
    goal_orientation_status = Column(Integer, default=0)

    svm_label = Column(String, nullable=True)
    verified_label = Column(String, nullable=True)

# ----------- Events Model -----------
class Event(BaseEvents):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer)
    event_type = Column(String)
    value = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Group(BaseGroups):
    __tablename__ = "groups"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    topic = Column(String)

class GroupMember(BaseGroups):
    __tablename__ = "group_members"
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey('groups.id'))
    user_id = Column(Integer)  


class GroupCreate(BaseModel):
    name: str
    topic: str
    member_ids: List[int]

    class Config:
        schema_extra = {
            "example": {
                "name": "Lerngruppe 1",
                "topic": "Wirtschaftsinformatik",
                "member_ids": [1, 2, 3]
            }
        }

class GroupOut(BaseModel):
    id: int
    name: str
    topic: str
    members: List[int]        

SessionLocalGroups = sessionmaker(bind=engine_groups)
BaseGroups.metadata.create_all(bind=engine_groups)


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

# ----------- Hilfsfunktionen für Matching etc. -----------
def get_vector(profile, verified=False):
    if verified:
        return [
            profile.extraversion_verified,
            profile.agreeableness_verified,
            profile.conscientiousness_verified,
            profile.emotional_stability_verified,
            profile.openness_verified,
            profile.team_orientation_verified,
            profile.motivation_verified,
            profile.communication_verified,
            profile.cohesion_verified,
            profile.goal_orientation_verified,
        ]
    else:
        return [
            profile.extraversion,
            profile.agreeableness,
            profile.conscientiousness,
            profile.emotional_stability,
            profile.openness,
            profile.team_orientation,
            profile.motivation,
            profile.communication,
            profile.cohesion,
            profile.goal_orientation,
        ]

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

def human_score(score):
    if score >= 0.85:
        return "Sehr hohe Übereinstimmung"
    elif score >= 0.7:
        return "Hohe Übereinstimmung"
    elif score >= 0.5:
        return "Mittelmäßig passend"
    elif score >= 0.3:
        return "Geringe Übereinstimmung"
    else:
        return "Sehr geringe Übereinstimmung"

# ----------- FastAPI Setup -----------
app = FastAPI(title="Matching-API")
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ---------- Pydantic-Schemas für API ----------
class ProfileBase(BaseModel):
    username: str
    first_name: str
    last_name: str
    email: str
    university: str
    course_of_study: str
    extraversion: int
    agreeableness: int
    conscientiousness: int
    emotional_stability: int
    openness: int
    team_orientation: int
    motivation: int
    communication: int
    cohesion: int
    goal_orientation: int

class ProfileOut(ProfileBase):
    id: int
    svm_label: Optional[str]
    verified_label: Optional[str]

# ---------- NEU: Profil anlegen ----------
@app.post("/profile", response_model=ProfileOut)
def create_profile(profile: ProfileBase):
    db = SessionLocalProfiles()
    # Check for unique fields
    if db.query(Profile).filter(Profile.username == profile.username).first():
        db.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(Profile).filter(Profile.email == profile.email).first():
        db.close()
        raise HTTPException(status_code=400, detail="Email already exists")
    p = Profile(**profile.dict())
    params = [
        p.extraversion, p.agreeableness, p.conscientiousness, p.emotional_stability, p.openness,
        p.team_orientation, p.motivation, p.communication, p.cohesion, p.goal_orientation
    ]
    label = predict_label(params)
    p.svm_label = label
    db.add(p)
    db.commit()
    db.refresh(p)
    db.close()
    return p

@app.post("/groups/", response_model=GroupOut)
def create_group(group: GroupCreate):
    db = SessionLocalGroups()
    # Optional: Prüfe auf doppelten Gruppennamen
    if db.query(Group).filter(Group.name == group.name).first():
        db.close()
        raise HTTPException(status_code=400, detail="Group name already exists")
    group_obj = Group(name=group.name, topic=group.topic)
    db.add(group_obj)
    db.commit()
    db.refresh(group_obj)
    group_id = group_obj.id  # <- id sichern solange Session offen

    for user_id in group.member_ids:
        db.add(GroupMember(group_id=group_id, user_id=user_id))
    db.commit()
    db.close()

    # id jetzt aus lokalem Wert nehmen, nicht mehr aus group_obj
    return GroupOut(
        id=group_id,
        name=group.name,
        topic=group.topic,
        members=group.member_ids
    )






@app.post("/group/{group_id}/add_member/")
def add_member(group_id: int, user_id: int):
    db = SessionLocalGroups()
    db.add(GroupMember(group_id=group_id, user_id=user_id))
    db.commit()
    db.close()
    return {"group_id": group_id, "user_id": user_id, "status": "added"}

# ---------- NEU: Profil aktualisieren ----------
@app.put("/profile/{profile_id}", response_model=ProfileOut)
def update_profile(profile_id: int, profile: ProfileBase):
    db = SessionLocalProfiles()
    p = db.query(Profile).filter(Profile.id == profile_id).first()
    if not p:
        db.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    # Unique checks
    if profile.username != p.username and db.query(Profile).filter(Profile.username == profile.username).first():
        db.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    if profile.email != p.email and db.query(Profile).filter(Profile.email == profile.email).first():
        db.close()
        raise HTTPException(status_code=400, detail="Email already exists")
    for field, value in profile.dict().items():
        setattr(p, field, value)
    params = [
        p.extraversion, p.agreeableness, p.conscientiousness, p.emotional_stability, p.openness,
        p.team_orientation, p.motivation, p.communication, p.cohesion, p.goal_orientation
    ]
    p.svm_label = predict_label(params)
    db.commit()
    db.refresh(p)
    db.close()
    return p

# ---------- NEU: RL-Label nachtragen ----------
@app.put("/profile/{profile_id}/verified_label")
def set_verified_label(profile_id: int, label: str):
    db = SessionLocalProfiles()
    p = db.query(Profile).filter(Profile.id == profile_id).first()
    if not p:
        db.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    p.verified_label = label
    db.commit()
    db.close()
    return {"status": f"verified_label '{label}' gesetzt."}


# ----------- Endpunkt: Profilübersicht (inkl. Fremdeinschätzung) -----------
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
        "id": p.id,
        "name": f"{p.first_name} {p.last_name}",
        "username": p.username,
        "email": p.email,
        "university": p.university,
        "course_of_study": p.course_of_study,
        "persönlichkeitsprofil": [
            {
                "eigenschaft": "Extraversion",
                "selbst": p.extraversion,
                "fremd": p.extraversion_verified,
                "status": status_str(p.extraversion_status)
            },
            {
                "eigenschaft": "Verträglichkeit",
                "selbst": p.agreeableness,
                "fremd": p.agreeableness_verified,
                "status": status_str(p.agreeableness_status)
            },
            {
                "eigenschaft": "Gewissenhaftigkeit",
                "selbst": p.conscientiousness,
                "fremd": p.conscientiousness_verified,
                "status": status_str(p.conscientiousness_status)
            },
            {
                "eigenschaft": "Emotionale Stabilität",
                "selbst": p.emotional_stability,
                "fremd": p.emotional_stability_verified,
                "status": status_str(p.emotional_stability_status)
            },
            {
                "eigenschaft": "Offenheit für Erfahrungen",
                "selbst": p.openness,
                "fremd": p.openness_verified,
                "status": status_str(p.openness_status)
            },
            {
                "eigenschaft": "Teamorientierung",
                "selbst": p.team_orientation,
                "fremd": p.team_orientation_verified,
                "status": status_str(p.team_orientation_status)
            },
            {
                "eigenschaft": "Motivation",
                "selbst": p.motivation,
                "fremd": p.motivation_verified,
                "status": status_str(p.motivation_status)
            },
            {
                "eigenschaft": "Kommunikationsfähigkeit",
                "selbst": p.communication,
                "fremd": p.communication_verified,
                "status": status_str(p.communication_status)
            },
            {
                "eigenschaft": "Kohäsion/Zusammenhalt",
                "selbst": p.cohesion,
                "fremd": p.cohesion_verified,
                "status": status_str(p.cohesion_status)
            },
            {
                "eigenschaft": "Zielorientierung",
                "selbst": p.goal_orientation,
                "fremd": p.goal_orientation_verified,
                "status": status_str(p.goal_orientation_status)
            },
        ],
        "svm_label": p.svm_label,
        "verified_label": p.verified_label
    }

# ----------- Endpunkt: Alle Profile -----------
@app.get("/profiles")
def list_profiles():
    db = SessionLocalProfiles()
    profiles = db.query(Profile).all()
    db.close()
    return [{"id": p.id, "username": p.username, "name": f"{p.first_name} {p.last_name}"} for p in profiles]

# ----------- Endpunkt: Matching Score -----------
from fastapi import HTTPException

@app.get("/match_score_advanced")
def match_score_advanced(profile_id_1: int, profile_id_2: int):
    db = SessionLocalProfiles()
    p1 = db.query(Profile).filter(Profile.id == profile_id_1).first()
    p2 = db.query(Profile).filter(Profile.id == profile_id_2).first()
    db.close()
    if not p1 or not p2:
        raise HTTPException(status_code=404, detail="Profile not found")

    def get_score_and_source(obj, attr, verified_attr):
        val = getattr(obj, verified_attr)
        if val is not None:
            return val, "verified"
        return getattr(obj, attr), "self"

    homogen_attrs = [
        ("conscientiousness", "conscientiousness_verified"),
        ("emotional_stability", "emotional_stability_verified"),
    ]
    heterogen_attrs = [
        ("extraversion", "extraversion_verified"),
        ("openness", "openness_verified"),
        ("agreeableness", "agreeableness_verified"),
        ("team_orientation", "team_orientation_verified"),
        ("communication", "communication_verified"),
    ]

    scores = []
    details = []
    used_sources = []  # Speichert die Source-Info für jedes Attribut

    for attr, vattr in homogen_attrs:
        v1, src1 = get_score_and_source(p1, attr, vattr)
        v2, src2 = get_score_and_source(p2, attr, vattr)
        diff = abs(v1 - v2)
        part_score = 1 - (diff / 4)
        scores.append(part_score)
        details.append({
            "dimension": attr,
            "type": "homogen",
            "score": round(part_score, 2),
            "source_1": src1,
            "source_2": src2
        })
        used_sources.append((src1, src2))

    for attr, vattr in heterogen_attrs:
        v1, src1 = get_score_and_source(p1, attr, vattr)
        v2, src2 = get_score_and_source(p2, attr, vattr)
        diff = abs(v1 - v2)
        part_score = diff / 4
        scores.append(part_score)
        details.append({
            "dimension": attr,
            "type": "heterogen",
            "score": round(part_score, 2),
            "source_1": src1,
            "source_2": src2
        })
        used_sources.append((src1, src2))

    overall = round(sum(scores) / len(scores), 2)

    # Erkläre das Source-Muster auf Gesamt-Ebene
    src_types = set([f"{s1}_vs_{s2}" for s1, s2 in used_sources])
    if len(src_types) == 1:
        overall_source = list(src_types)[0]
    else:
        overall_source = ", ".join(sorted(src_types))

    # Bewertungstext
    if overall > 0.75:
        msg = "Sehr gute Teamkompatibilität (ausbalancierte Mischung aus Gemeinsamkeit und Unterschiedlichkeit)."
    elif overall > 0.5:
        msg = "Gute Teamkompatibilität."
    elif overall > 0.3:
        msg = "Mittlere Teamkompatibilität."
    else:
        msg = "Geringe Teamkompatibilität."

    return {
        "profile_id_1": profile_id_1,
        "profile_id_2": profile_id_2,
        "matching_score": overall,
        "matching_details": details,
        "sources": overall_source,  # z.B. "verified_vs_verified", "self_vs_verified", "self_vs_self"
        "summary": msg,
        "basis": "wissenschaftliches Matching (Big Five, Bellhäuser et al.)"
    }

@app.get("/groups/{group_id}", response_model=GroupOut)
def get_group(group_id: int):
    db = SessionLocalGroups()
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        db.close()
        raise HTTPException(status_code=404, detail="Group not found")
    member_ids = [gm.user_id for gm in db.query(GroupMember).filter(GroupMember.group_id == group_id).all()]
    db.close()
    return GroupOut(
        id=group.id,
        name=group.name,
        topic=group.topic,
        members=member_ids
    )


from fastapi import Query

@app.get("/group_match_score")
def group_match_score(candidate_id: int, group_ids: list[int] = Query(...)):
    db = SessionLocalProfiles()
    candidate = db.query(Profile).filter(Profile.id == candidate_id).first()
    group = db.query(Profile).filter(Profile.id.in_(group_ids)).all()
    db.close()
    if not candidate or not group:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Helper
    def get_score_and_source(obj, attr, verified_attr):
        val = getattr(obj, verified_attr, None)
        if val is not None:
            return val, "verified"
        return getattr(obj, attr, None), "self"

    homogen_attrs = [
        ("conscientiousness", "conscientiousness_verified"),
        ("emotional_stability", "emotional_stability_verified"),
    ]
    heterogen_attrs = [
        ("extraversion", "extraversion_verified"),
        ("openness", "openness_verified"),
        ("agreeableness", "agreeableness_verified"),
    ]

    pair_scores = []
    pair_details = []
    for member in group:
        scores = []
        for attr, vattr in homogen_attrs:
            v1, _ = get_score_and_source(candidate, attr, vattr)
            v2, _ = get_score_and_source(member, attr, vattr)
            if v1 is None or v2 is None:
                continue  # Fehlerfall oder fehlende Werte abfangen
            diff = abs(v1 - v2)
            scores.append(1 - (diff / 4))
        for attr, vattr in heterogen_attrs:
            v1, _ = get_score_and_source(candidate, attr, vattr)
            v2, _ = get_score_and_source(member, attr, vattr)
            if v1 is None or v2 is None:
                continue
            diff = abs(v1 - v2)
            scores.append(diff / 4)
        if scores:
            mean_score = sum(scores) / len(scores)
        else:
            mean_score = 0
        pair_scores.append(mean_score)
        pair_details.append({"member_id": member.id, "score": round(mean_score, 2)})
    
    # Alpha-Check
    alpha_count = sum([
        1 for m in group
        if get_score_and_source(m, "extraversion", "extraversion_verified")[0] is not None
        and get_score_and_source(m, "extraversion", "extraversion_verified")[0] >= 5
    ])
    candidate_alpha = get_score_and_source(candidate, "extraversion", "extraversion_verified")[0] is not None and get_score_and_source(candidate, "extraversion", "extraversion_verified")[0] >= 5
    alpha_kritisch = (alpha_count + int(candidate_alpha)) > 1

    group_score = min(pair_scores) if pair_scores else 0
    details = {
        "pair_scores": pair_details,
        "alpha_count_in_group": alpha_count,
        "candidate_is_alpha": candidate_alpha,
        "alpha_critical": alpha_kritisch
    }
    if alpha_kritisch:
        msg = "Kritisch: Zu viele dominante/extrovertierte Personen in der Gruppe!"
    elif group_score > 0.75:
        msg = "Sehr gute Passung zur Gruppe."
    elif group_score > 0.5:
        msg = "Gute Passung zur Gruppe."
    elif group_score > 0.3:
        msg = "Mittelmäßige Passung."
    else:
        msg = "Schwache Passung."

    return {
        "candidate_id": candidate_id,
        "group_ids": group_ids,
        "group_score": round(group_score, 2),
        "details": details,
        "summary": msg,
        "basis": "wissenschaftliches Gruppenmatching"
    }

@app.get("/group_match_score_by_groupid")
def group_match_score_by_groupid(candidate_id: int, group_id: int):
    # Hole Gruppenmitglieder-IDs aus der Datenbank (über group_members-Tabelle!)
    db_g = SessionLocalGroups()
    group = db_g.query(Group).filter(Group.id == group_id).first()
    if not group:
        db_g.close()
        raise HTTPException(status_code=404, detail="Group not found")
    member_ids = [gm.user_id for gm in db_g.query(GroupMember).filter(GroupMember.group_id == group_id).all()]
    db_g.close()
    # Verwende den bestehenden Matching-Code, übergib die member_ids
    return group_match_score(candidate_id=candidate_id, group_ids=member_ids)



# ----------- Dummy-Event-Generator -----------
@app.post("/event/dummy/{profile_id}")
def create_dummy_event(profile_id: int):
    db_e = SessionLocalEvents()
    # Beispiel: für jedes Attribut 1 Event mit zufälligem Wert 1-5
    for attr in [
        "extraversion", "agreeableness", "conscientiousness", "emotional_stability",
        "openness", "team_orientation", "motivation", "communication", "cohesion", "goal_orientation"
    ]:
        value = random.randint(1, 5)
        db_e.add(Event(profile_id=profile_id, event_type=attr, value=value))
    db_e.commit()
    db_e.close()
    return {"status": "Dummy-Events erzeugt"}

# ----------- Events abrufen -----------
@app.get("/event/{profile_id}")
def get_events(profile_id: int):
    db_e = SessionLocalEvents()
    events = db_e.query(Event).filter(Event.profile_id == profile_id).all()
    db_e.close()
    return [
        {"event_type": e.event_type, "value": e.value, "timestamp": e.timestamp.isoformat()}
        for e in events

    ]
