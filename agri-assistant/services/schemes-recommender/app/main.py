import os
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

# ---------- Config ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@postgres:5432/agri"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

app = FastAPI(title="Schemes Recommender", version="1.0.0")


# ---------- Models ----------
class Scheme(BaseModel):
    id: uuid.UUID                               # <— accept UUID from DB
    title: str
    description: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    crop: Optional[str] = None
    implementing_agency: Optional[str] = None
    url: Optional[str] = None
    benefit_type: Optional[str] = None
    benefit_details: Optional[str] = None
    eligibility: Dict[str, Any] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    active: bool = True

    # runtime score we compute for ranking (not stored)
    score: float = 0.0


# ---------- Helpers ----------
def _normalize(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s if s else None


def _score_row(row: Dict[str, Any],
               land_size_ha: Optional[float],
               annual_income: Optional[float],
               category: Optional[str],
               irrigation: Optional[str]) -> float:
    """
    Simple heuristic score to rank schemes for a given farmer profile.
    You can tune these weights later.
    """
    score = 0.0

    # small farmer boost
    if category and isinstance(row.get("eligibility"), dict):
        ft = str(row["eligibility"].get("farmer_type", "")).lower()
        if category.lower() in (ft, "all"):
            score += 1.0

    # income cap check, if present
    if annual_income is not None and isinstance(row.get("eligibility"), dict):
        cap = row["eligibility"].get("income_cap")
        try:
            if cap is not None:
                cap_val = float(cap)
                if annual_income <= cap_val:
                    score += 1.0
                else:
                    score -= 0.5
        except Exception:
            pass

    # irrigation match keyword
    if irrigation and isinstance(row.get("keywords"), list):
        kws = [str(k).lower() for k in row["keywords"]]
        if irrigation.lower() in kws:
            score += 0.5

    # land size hint (very rough)
    if land_size_ha is not None and isinstance(row.get("eligibility"), dict):
        max_land = row["eligibility"].get("land_holding_max")
        try:
            if max_land is not None:
                if float(land_size_ha) <= float(max_land):
                    score += 0.5
                else:
                    score -= 0.25
        except Exception:
            pass

    # benefit types are more impactful
    bt = (row.get("benefit_type") or "").lower()
    if bt in ("subsidy", "cash", "insurance"):
        score += 0.5

    return score


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/schemes", response_model=List[Scheme])
def get_schemes(
    country: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    crop: Optional[str] = Query(None),
    land_size_ha: Optional[float] = Query(None),
    annual_income: Optional[float] = Query(None),
    category: Optional[str] = Query(None),         # e.g., small/marginal/general
    irrigation: Optional[str] = Query(None),       # e.g., rainfed/irrigated
    limit: int = Query(50, ge=1, le=200),
):
    """
    Filter schemes by geo/crop and rank them by simple heuristics using
    farmer profile hints. Returns top-N results.
    """
    country = _normalize(country)
    state = _normalize(state)
    district = _normalize(district)
    crop = _normalize(crop)

    where = ["active = TRUE"]
    params: Dict[str, Any] = {}

    if country:
        where.append("(country = :country OR country = 'ALL')")
        params["country"] = country.upper()
    if state:
        where.append("(state = :state OR state = 'ALL')")
        params["state"] = state
    if district:
        where.append("(district = :district OR district IS NULL)")
        params["district"] = district
    if crop:
        where.append("(crop = :crop OR crop = 'ALL')")
        params["crop"] = crop

    where_sql = " AND ".join(where) if where else "TRUE"

    # NOTE:
    # - We select native types: UUID (id), jsonb (eligibility), text[] (keywords).
    # - psycopg2 adapts these to Python types: uuid.UUID, dict, list[str].
    sql = text(f"""
        SELECT
          id,
          title,
          description,
          country,
          state,
          district,
          crop,
          implementing_agency,
          url,
          benefit_type,
          benefit_details,
          eligibility,
          keywords,
          active
        FROM gov_schemes
        WHERE {where_sql}
        LIMIT :limit
    """)

    params["limit"] = limit

    rows: List[Scheme] = []
    with engine.begin() as conn:
        res = conn.execute(sql, params)
        for r in res.mappings():  # returns dict-like rows
            row = dict(r)

            # Score per farmer profile
            score = _score_row(
                row,
                land_size_ha=land_size_ha,
                annual_income=annual_income,
                category=category,
                irrigation=irrigation,
            )

            # Build model (id is uuid.UUID -> OK)
            sch = Scheme(
                id=row["id"],
                title=row["title"],
                description=row.get("description"),
                country=row.get("country"),
                state=row.get("state"),
                district=row.get("district"),
                crop=row.get("crop"),
                implementing_agency=row.get("implementing_agency"),
                url=row.get("url"),
                benefit_type=row.get("benefit_type"),
                benefit_details=row.get("benefit_details"),
                eligibility=row.get("eligibility") or {},
                keywords=row.get("keywords") or [],
                active=bool(row.get("active", True)),
                score=score,
            )
            rows.append(sch)

    # Sort by score desc, then by title asc to stabilize
    rows.sort(key=lambda x: (-x.score, x.title.lower()))
    return rows[:limit]
