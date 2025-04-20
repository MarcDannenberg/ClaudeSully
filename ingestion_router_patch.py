# 1. Remove any existing /ingest_document route in sully_api.py
# 2. Add this import near the top
from sully_ingestion_master import router as ingestion_router

# 3. At the bottom of sully_api.py, register the new route
app.include_router(ingestion_router)