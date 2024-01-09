import datetime
import functools
import importlib
import json
import logging
import sqlite3
from typing import Optional, Type
import typing
import html2text
import llm
import numpy as np
import re

import requests
from openai import OpenAI
import instructor

from fossil_mastodon import config
if typing.TYPE_CHECKING:
    from fossil_mastodon.algorithm import base
import os
from pydantic import BaseModel

from loguru import logger

instructor_client = instructor.patch(OpenAI())

@functools.cache
def create_database():
    if os.path.exists(config.DATABASE_PATH):
        return

    print("Creating database")
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        c = conn.cursor()

        # Create the toots table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS toots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                author TEXT,
                url TEXT,
                created_at DATETIME,
                embedding BLOB,
                orig_json TEXT,
                cluster TEXT  -- Added cluster column
            )
        ''')

        conn.commit()


Base = declarative_base()

class Toot(Base):
    __tablename__ = 'toots'

    instance_url = Column(String, primary_key=True)
    id = Column(Integer, primary_key=True)
    content = Column(String)
    author = Column(String)
    url = Column(String)
    created_at = Column(DateTime)
    embedding = Column(LargeBinary)
    orig_json = Column(String)
    cluster = Column(String)

    def from_dict(self, data):
        import json

        if data.get("reblog"):
            return self.from_dict(data["reblog"])

        return self(
            content=data.get("content"),
            author=data.get("account", {}).get("acct"),
            url=data.get("url"),
            created_at=datetime.datetime.strptime(data.get("created_at"), "%Y-%m-%dT%H:%M:%S.%fZ"),
            orig_json=json.dumps(data),
        )

    def get_toots_since(self, since: datetime.datetime):
        return self.query.filter(self.created_at > since).all()


engine = create_engine('sqlite:///fossil_mastodon.db')
Base.metadata.create_all(engine)

def get_toots_since(since: datetime.datetime):
    assert isinstance(since, datetime.datetime), type(since)
    create_database()
    download_timeline(since)
    return Toot.get_toots_since(since)


def download_timeline(since: datetime.datetime):
    last_date = Toot.get_latest_date()
    logger.info(f"last toot date: {last_date}")
    last_date = last_date or since
    earliest_date = None
    buffer: list[Toot] = []
    last_id = ""
    curr_url = f"{config.MASTO_BASE}/api/v1/timelines/home?limit=40"
    import json as JSON
    while not earliest_date or earliest_date > last_date:
        response = requests.get(curr_url, headers=config.headers())
        response.raise_for_status()
        json = response.json()
        if not json:
            logger.info("No more toots")
            break
        if len(json) > 1:
            last_id = json[-1]["id"]
        logger.info(f"Got {len(json)} toots; earliest={earliest_date.isoformat() if earliest_date else None}, last_id={last_id}")
        for toot_dict in json:
            toot = Toot.from_dict(toot_dict)
            earliest_date = toot.created_at if not earliest_date else min(earliest_date, datetime.datetime.strptime(toot_dict["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"))
            buffer.append(toot)

        if "next" in response.links:
            curr_url = response.links["next"]["url"]
        else:
            break
    logger.info(f"done with toots; earliest={earliest_date.isoformat() if earliest_date else None}, last_date: {last_date.isoformat() if last_date else None}")

    page_size = 50
    if len(buffer) > 0:
        num_pages = len(buffer) // page_size + 1
    else:
        num_pages = 0
    for page in range(num_pages):
        start_index = page * page_size
        end_index = start_index + page_size
        page_toots = buffer[start_index:end_index]

        tags = _create_tags(page_toots)
        _create_embeddings(page_toots)
        with sqlite3.connect(config.DATABASE_PATH) as conn:
            for toot in page_toots:
                toot.save(init_conn=conn)



def _create_embeddings(toots: list[Toot]):
    # Convert the list of toots to a single string
    toots = [t for t in toots if t.content]

    # Call the llm embedding API to create embeddings
    emb_model = llm.get_embedding_model(config.EMBEDDING_MODEL.name)
    embeddings = list(emb_model.embed_batch([html2text.html2text(t.content) for t in toots]))

    # Extract the embeddings from the API response
    print(f"got {len(embeddings)} embeddings")
    for i, toot in enumerate(toots):
        toot.embedding = np.array(embeddings[i])

    # Return the embeddings
    return toots

class TootTags(BaseModel):
    topic: str
    is_polemical: bool
    is_positive_valence: bool
    is_negative_valence: bool
    is_neutral_valence: bool
    is_emotional_language: bool
    is_political: bool
    is_academic: bool
    is_news: bool
    is_cringe: bool
    is_based: bool
    is_israel_palestine: bool
    is_joke: bool
    is_art: bool
    is_trolling: bool
    is_opinion: bool
    is_academic: bool
    is_informative: bool

def cleanup_toot_text(toot):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', toot.content)
    return f'{toot.display_name} posted: {text}'


def _create_tags(toots: list[Toot]):
    toot_tags = []
    for toot in toots:
        cleaned_text = cleanup_toot_text(toot)
        toot_tagged = instructor_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_model=TootTags,
                messages = [
                    {"role": "user", "content": f"Classify the following text: {cleaned_text}" }
                    ])
        toot_tags.append(toot_tagged)
    return toot_tags


@functools.lru_cache()
def _create_session_table():
    create_database()
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        c = conn.cursor()

        # Create the toots table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                algorithm_spec TEXT,
                algorithm BLOB,
                ui_settings TEXT
            )
        ''')

        conn.commit()


class Session(BaseModel):
    id: str
    algorithm_spec: str | None = None
    algorithm: bytes | None = None
    ui_settings: str | None = None

    def set_ui_settings(self, ui_settings: dict[str, str]):
        self.ui_settings = json.dumps(ui_settings)
        self.save()

    def get_ui_settings(self) -> dict[str, str]:
        return json.loads(self.ui_settings or "{}")

    def get_algorithm_type(self) -> Type["base.BaseAlgorithm"] | None:
        spec = json.loads(self.algorithm_spec) if self.algorithm_spec else {}
        if "module" in spec and "class_name" in spec:
            mod = importlib.import_module(spec["module"])
            return getattr(mod, spec["class_name"])
        return None

    def set_algorithm_by_name(self, name: str) -> Type["base.BaseAlgorithm"] | None:
        from fossil_mastodon.algorithm import base
        algo = next((algo for algo in base.get_algorithms() if algo.get_name() == name), None)
        self.algorithm_spec = json.dumps({
            "module": algo.__module__,
            "class_name": algo.__name__,
            "kwargs": {},
        })
        self.algorithm = None
        self.save()
        return algo

    @classmethod
    def get_by_id(cls, id: str) -> Optional["Session"]:
        create_database()
        _create_session_table()
        with sqlite3.connect(config.DATABASE_PATH) as conn:
            c = conn.cursor()

            c.execute('''
                SELECT id, algorithm_spec, algorithm, ui_settings FROM sessions WHERE id = ?
            ''', (id,))

            row = c.fetchone()
            if row:
                session = cls(
                    id=row[0],
                    algorithm_spec=row[1],
                    algorithm=row[2],
                    ui_settings=row[3],
                )
                return session
            return None

    @classmethod
    def create(cls) -> "Session":
        import uuid
        return cls(id=str(uuid.uuid4()).replace("-", ""))

    def save(self, init_conn: sqlite3.Connection | None = None) -> bool:
        _create_session_table()
        try:
            if init_conn is None:
                conn = sqlite3.connect(config.DATABASE_PATH)
            else:
                conn = init_conn
            create_database()
            c = conn.cursor()

            c.execute('''
                INSERT INTO sessions (id, algorithm_spec, algorithm, ui_settings)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE 
                    SET algorithm_spec = excluded.algorithm_spec
                      , algorithm = excluded.algorithm
                      , ui_settings = excluded.ui_settings
            ''', (self.id, self.algorithm_spec, self.algorithm, self.ui_settings))

            if init_conn is None:
                conn.commit()
        except:
            conn.rollback()
            raise
        return True
