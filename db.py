"""
db.py: save and read community uploads.

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence

If a MongoDB connection string is set (Streamlit secret MONGO_URI or the
environment variable of the same name), uploads go to MongoDB Atlas.
If not, or if the database can't be reached, the app keeps working and
stores uploads in memory until it restarts.
"""
import datetime as dt
import os

import pandas as pd

DB_NAME, COLLECTION = "fasalscope", "uploads"


class MemoryStore:
    kind = "Temporary memory"

    def __init__(self):
        self.rows = []

    def add(self, record):
        self.rows.append(dict(record))

    def all(self):
        return pd.DataFrame(self.rows)


class MongoStore:
    kind = "MongoDB Atlas"

    def __init__(self, uri, client=None):
        if client is None:
            from pymongo import MongoClient
            client = MongoClient(uri, serverSelectionTimeoutMS=4000)
            client.admin.command("ping")  # fail fast if unreachable
        self.col = client[DB_NAME][COLLECTION]

    def add(self, record):
        self.col.insert_one(dict(record))

    def all(self, limit=2000):
        rows = list(self.col.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
        return pd.DataFrame(rows)


def get_store(secrets=None):
    """Pick MongoDB if configured and reachable, otherwise memory. Returns (store, message)."""
    uri = None
    try:
        if secrets is not None and "MONGO_URI" in secrets:
            uri = secrets["MONGO_URI"]
    except Exception:
        uri = None
    uri = uri or os.environ.get("MONGO_URI")
    if not uri:
        return MemoryStore(), "No database configured: uploads are kept until the app restarts."
    try:
        return MongoStore(uri), "Connected to MongoDB Atlas."
    except Exception as e:  # wrong password, network blocked, Atlas paused…
        return MemoryStore(), f"Couldn't reach MongoDB ({type(e).__name__}); using temporary memory."


def make_record(name, feats, probs, verdict, thumb, user_label=None, unusual=False):
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "name": (name or "anonymous").strip()[:30],
        "user_label": user_label,
        "verdict": verdict,
        "unusual": bool(unusual),
        **{f: feats[f] for f in feats},
        **{"p_" + k.lower().replace(" ", "_"): round(v, 4) for k, v in probs.items()},
        "thumb": thumb,
    }
