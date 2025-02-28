import json
import pickle
import time
import psycopg2
from dotenv import load_dotenv
import os
import logging
from nn.jetson.test_inference import infer_test

# %% imports
import psycopg2.extras


# %% configure logger
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# %% connect to db
load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    sslmode="require",
)

# %% EDGE side - get new edge_measurements
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# load cutoff_timestamp
if os.path.exists("cutoff_ts.pkl"):
    with open("cutoff_ts.pkl", "rb") as f:
        cutoff_timestamp = pickle.load(f)
else:
    cutoff_timestamp = "'1970-01-01 00:00:00.0'"

tryN = 0
while True:
    cur.execute(
        """
        SELECT * FROM network_architectures 
        WHERE created > timestamp %s 
        ORDER BY created ASC
        LIMIT 10
        """,
        (cutoff_timestamp,),
    )

    network_architectures = cur.fetchall()

    if len(network_architectures) == 0:
        if tryN % 30 == 0:
            logging.info(
                f"No new network_architectures found with cutoff `{cutoff_timestamp}`, retrying silently every 60s 30 times..."
            )
        tryN += 1
        time.sleep(60)
        continue

    tryN = 0

    # %% run inference on EDGE
    devices = [(8, "Jetson AGX Orin 64GB")]
    batch_sizes = [1, 2, 4, 8, 16]

    for naI, na in enumerate(network_architectures):
        logging.info(
            f"Running inference {naI+1}/{len(network_architectures)} with ID {na['id']}"
        )
        for device_id, device in devices:
            for batch_size in batch_sizes:
                logging.info(
                    f"Running inference on {device} with batch_size={batch_size}"
                )
                res = infer_test(
                    model_name=na["name"],
                    batch_size=batch_size,
                    hyperparameters=na["hyperparameters"],
                )

                cur.execute(
                    """
                    INSERT INTO edge_measurements 
                        (network_architecture_id, device_id, batch_size, latency_ms, results) 
                    VALUES 
                        (%s, %s, %s, %s, %s)
                    """,
                    (
                        na["id"],
                        device_id,
                        batch_size,
                        res["avg_inference_time"],
                        json.dumps(res),
                    ),
                )
                conn.commit()
                cutoff_timestamp = na["created"]
                # save cutoff_timestamp
                with open("cutoff_ts.pkl", "wb") as f:
                    pickle.dump(cutoff_timestamp, f)
                logging.info(f"Committed edge_measurement (BS={batch_size})")
        logging.info(
            f"All edge_measurements tested for network_architecture_id {na['id']}"
        )

# %% close connection
conn.commit()  # NOTE: reopen conn as least as possible
cur.close()
conn.close()
