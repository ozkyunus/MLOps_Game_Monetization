"""
Load real datasets to Postgres.

Real backbone (Marketing Freemium Game Data - Kaggle):
  - raw_user_source        (15,155 rows) — installs + attribution
  - raw_purchases          (1,427 rows) — transactions + LTV
  - raw_ad_spend           (60 rows) — UA campaigns

Real event-level reference (Firebase Public Project — Flood It!):
  - raw_flood_it_users     (749 rows) — D7 aggregated user features
"""
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))

# 1. user_source
print("Loading user_source...")
df = pd.read_csv(
    "data/raw/user_source.csv",
    parse_dates=["install_date"],
)
df.to_sql("raw_user_source", engine, if_exists="replace", index=False)
print(f"  ✓ raw_user_source: {len(df)} rows")

# 2. purchases (tab-separated)
print("Loading purchases...")
df = pd.read_csv(
    "data/raw/purchases.tsv",
    sep="\t",
    parse_dates=["date"],
)
df.to_sql("raw_purchases", engine, if_exists="replace", index=False)
print(f"  ✓ raw_purchases: {len(df)} rows")

# 3. ad_spend
print("Loading ad_spend...")
df = pd.read_csv(
    "data/raw/ad_spend.csv",
    parse_dates=["date"],
)
df.to_sql("raw_ad_spend", engine, if_exists="replace", index=False)
print(f"  ✓ raw_ad_spend: {len(df)} rows")

# 4. Flood It! D7 user features (Firebase public dataset)
print("Loading flood_it_users...")
df = pd.read_csv("data/raw/flood_it_users_d7.csv")
# install_date is YYYYMMDD int — convert to datetime
df["install_date"] = pd.to_datetime(df["install_date"].astype(str), format="%Y%m%d")
df.to_sql("raw_flood_it_users", engine, if_exists="replace", index=False)
print(f"  ✓ raw_flood_it_users: {len(df)} rows")

print("\nAll datasets loaded successfully.")