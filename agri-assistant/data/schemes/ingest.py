import csv, json, uuid, os, re
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@127.0.0.1:5433/agri")

def norm(x):
    x = (x or "").strip()
    return None if x == "" else x

def to_country_code(val: str | None):
    if not val:
        return None
    v = val.strip().upper()
    # Normalize common names -> ISO-ish codes you query with
    if v in ("INDIA", "BHARAT", "REPUBLIC OF INDIA"):
        return "IN"
    return v

def parse_keywords(s):
    return [k.strip() for k in (s or "").split("|") if k.strip()]

def main(csv_path):
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with engine.begin() as conn:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                # Parse and DUMP JSON so psycopg2 sees a string (we'll cast to jsonb in SQL)
                try:
                    elig_obj = json.loads(row.get("eligibility_json") or "{}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad eligibility_json in row with title='{row.get('title')}': {e}")

                eligibility_json_str = json.dumps(elig_obj, ensure_ascii=False)

                keywords_list = parse_keywords(row.get("keywords"))
                # psycopg2 can adapt Python list -> text[] if we CAST in SQL
                payload = {
                    "id": str(uuid.uuid4()),
                    "title": row["title"],
                    "description": norm(row.get("description")),
                    "country": to_country_code(norm(row.get("country"))),   # normalize
                    "state": norm(row.get("state")),
                    "district": norm(row.get("district")),
                    "crop": norm(row.get("crop")),
                    "implementing_agency": norm(row.get("implementing_agency")),
                    "url": norm(row.get("url")),
                    "benefit_type": norm(row.get("benefit_type")),
                    "benefit_details": norm(row.get("benefit_details")),
                    "eligibility": eligibility_json_str,   # STRING here; cast to jsonb in SQL
                    "keywords": keywords_list,             # list; cast to text[] in SQL
                }

                conn.execute(text("""
                    INSERT INTO gov_schemes
                      (id,title,description,country,state,district,crop,
                       implementing_agency,url,benefit_type,benefit_details,
                       eligibility,keywords,active)
                    VALUES
                      (:id,:title,:description,:country,:state,:district,:crop,
                       :implementing_agency,:url,:benefit_type,:benefit_details,
                       CAST(:eligibility AS jsonb), CAST(:keywords AS text[]), TRUE)
                    ON CONFLICT (id) DO NOTHING
                """), payload)

    print("Imported schemes from", csv_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data/schemes/ingest.py data/schemes/sample_schemes.csv")
        raise SystemExit(1)
    main(sys.argv[1])
