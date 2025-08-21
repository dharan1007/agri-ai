from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Boolean, JSON, ARRAY

class Base(DeclarativeBase): pass

class GovScheme(Base):
    __tablename__ = "gov_schemes"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str | None] = mapped_column(String, nullable=True)
    district: Mapped[str | None] = mapped_column(String, nullable=True)
    crop: Mapped[str | None] = mapped_column(String, nullable=True)
    implementing_agency: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    benefit_type: Mapped[str | None] = mapped_column(String, nullable=True)
    benefit_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    eligibility: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

