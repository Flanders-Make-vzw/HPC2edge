import json
import pickle
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

# taken from example script
load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    sslmode="require",
)

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# insert test architecture
cur.execute(
    """
    INSERT INTO network_architectures (name, hyperparameters)
    VALUES
    ('Swin3D', '{
        "name": "Swin3D",
        "in_chans": 1,
        "patch_size": [2, 2, 2],
        "embed_dim": 96,
        "depths": [1, 1, 1, 1],
        "num_heads": [3, 6, 12, 24],
        "window_size": [8, 7, 7],
        "mlp_ratio": 4.0,
        "num_classes": 400
        }')
    """,
)

cur.close()
conn.close()
