-- Enable UUID generator
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===============================
-- Core farmer data
-- ===============================
CREATE TABLE IF NOT EXISTS farmers (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  phone TEXT UNIQUE,
  name TEXT,
  language TEXT DEFAULT 'en',
  village TEXT,
  district TEXT,
  state TEXT,
  country TEXT,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  consent_data BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_farmers_geo ON farmers(country, state, district);
CREATE INDEX IF NOT EXISTS idx_farmers_loc ON farmers(lat, lon);

CREATE TABLE IF NOT EXISTS fields (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  farmer_id UUID NOT NULL REFERENCES farmers(id) ON DELETE CASCADE,
  area_ha NUMERIC(10,2),
  soil_type TEXT,
  irrigation TEXT
);
CREATE INDEX IF NOT EXISTS idx_fields_farmer ON fields(farmer_id);

CREATE TABLE IF NOT EXISTS images (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  farmer_id UUID REFERENCES farmers(id) ON DELETE SET NULL,
  field_id UUID REFERENCES fields(id)   ON DELETE SET NULL,
  s3_uri TEXT,
  mime TEXT,
  width INT,
  height INT,
  captured_at TIMESTAMP,
  label_status TEXT DEFAULT 'unlabeled'
);
CREATE INDEX IF NOT EXISTS idx_images_farmer ON images(farmer_id);
CREATE INDEX IF NOT EXISTS idx_images_field  ON images(field_id);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(label_status);

CREATE TABLE IF NOT EXISTS labels (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  disease TEXT,
  confidence REAL,
  source TEXT,
  annotator TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id);
CREATE INDEX IF NOT EXISTS idx_labels_disease ON labels(disease);

CREATE TABLE IF NOT EXISTS model_artifacts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  task TEXT,                 -- e.g., disease-classifier
  version TEXT,              -- semantic or timestamp version
  framework TEXT,            -- torch/onnx/tflite
  s3_uri TEXT,               -- where the artifact is stored
  metrics JSONB,             -- e.g., {"val_acc":0.92,"f1":0.91}
  registered_at TIMESTAMP DEFAULT now(),
  is_production BOOLEAN DEFAULT FALSE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_task_version ON model_artifacts(task, version);
CREATE INDEX IF NOT EXISTS idx_model_is_production ON model_artifacts(is_production);

-- ===============================
-- Enhanced Government Schemes
-- (what the recommender service expects)
-- ===============================
CREATE TABLE IF NOT EXISTS gov_schemes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title TEXT NOT NULL,
  description TEXT,
  country TEXT,                 -- e.g., IN
  state TEXT,                   -- 'ALL' or specific
  district TEXT,                -- NULL / 'ALL' / specific
  crop TEXT,                    -- 'ALL' or specific (wheat, rice, etc.)
  implementing_agency TEXT,
  url TEXT,
  benefit_type TEXT,            -- subsidy|credit|insurance|equipment|training|market|benefit
  benefit_details TEXT,         -- free text summary / amount notes
  eligibility JSONB,            -- structured rules: {"max_land_ha":2,"category":["small"], ...}
  keywords TEXT[],              -- quick search tags
  active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_gs_geo ON gov_schemes(country, state, district);
CREATE INDEX IF NOT EXISTS idx_gs_crop ON gov_schemes(crop);
CREATE INDEX IF NOT EXISTS idx_gs_active ON gov_schemes(active);

-- ===============================
-- Seed sample schemes (edit/expand later)
-- ===============================
INSERT INTO gov_schemes
(title, description, country, state, district, crop, implementing_agency, url, benefit_type, benefit_details, eligibility, keywords, active)
VALUES
-- Soil Health Card (nationwide)
(
  'Soil Health Card',
  'Free soil testing and fertilizer recommendations.',
  'IN', 'ALL', NULL, 'ALL',
  'GoI',
  'https://example.gov/soil',
  'benefit',
  'Free/discounted soil testing through designated labs.',
  '{"country":["IN"]}'::jsonb,
  ARRAY['soil','testing','fertility'],
  TRUE
),

-- Drip Irrigation Subsidy (Maharashtra example, rainfed/borewell focus)
(
  'Drip Irrigation Subsidy',
  'Subsidy support for micro/drip irrigation systems.',
  'IN', 'Maharashtra', NULL, 'ALL',
  'Dept. Agriculture Maharashtra',
  'https://example.gov/drip',
  'subsidy',
  'Up to ₹60,000 depending on area and crop.',
  '{"country":["IN"],"state":["Maharashtra"],"land_holding_max":5,"irrigation":["rainfed","borewell"]}'::jsonb,
  ARRAY['drip','micro','water'],
  TRUE
),

-- Seed Replacement Program (Punjab, wheat, small farmers)
(
  'Seed Replacement Program',
  'Certified seeds at subsidized rates for Rabi season.',
  'IN', 'Punjab', NULL, 'wheat',
  'Dept. Agriculture Punjab',
  'https://example.gov/seed',
  'subsidy',
  'Subsidized distribution of certified wheat seeds.',
  '{"country":["IN"],"state":["Punjab"],"farmer_type":["small"]}'::jsonb,
  ARRAY['seed','kit','rabi','wheat'],
  TRUE
)
ON CONFLICT DO NOTHING;
