import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os


def connect_database():
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
    return conn


def send_data(name, config):
    
    conn = connect_database()
    
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    import json

    hyperparameters = {
        "name": "Swin3D",
        "in_chans": 1,
        "patch_size": [config["patch_size_1"], config["patch_size_2"], config["patch_size_3"]],  # Patch size
        "embed_dim": config["embed_dim"],
        "depths": [config["depths_1"], config["depths_2"], config["depths_3"], config["depths_4"]],  # Depths of each Swin Transformer stage
        "num_heads": [config["num_heads_1"], config["num_heads_2"], config["num_heads_3"], config["num_heads_4"]],  # Number of attention heads of each stage
        "window_size": [8, 7, 7],  # Window size
        "mlp_ratio": config["mlp_ratio"],  # Ratio of MLP hidden dim to embedding dim
        "num_classes": 400  # Penultimate hidden dim size
    }

    # Convert the JSON object to a string
    hyperparameters_str = json.dumps(hyperparameters)

    # Execute the SQL query
    cur.execute(
        "INSERT INTO network_architectures (name, hyperparameters) VALUES (%s, %s)",
        (str(name), hyperparameters_str)
    )
    
    # Retrieve the id of the newly inserted row by its name
    cur.execute("SELECT id FROM network_architectures WHERE name = %s", (str(name),))
    
    output = cur.fetchall()
    model_id = output[0]['id']
        
    conn.commit() 
    cur.close()
    conn.close()
    
    print("Sending data succesful!")
    
    return model_id
    
def receive_data(model_id):
    
    conn = connect_database()
    
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    import json

    # Execute the SQL query
    cur.execute("SELECT * FROM edge_measurements WHERE network_architecture_id = %s and batch_size= 2", (int(model_id),))
    
    output = cur.fetchall()
        
    if output:
        #print(output[0]['latency_ms'])

        conn.commit() 
        cur.close()
        conn.close()

        print("Receiving data succesful!")

        return (output[0]['latency_ms']/16)
    else:
        return 100000

# %% functions
def get_evaluation_measure_id(cur, evaluation_measure_name):
    cur.execute(
        """
        SELECT id FROM evaluation_measure
        WHERE name = %s
        """,
        (evaluation_measure_name,),
    )
    evaluation_measure_ids = cur.fetchall()
    if len(evaluation_measure_ids) == 0:
        return

    evaluation_measure_id = evaluation_measure_ids[0]["id"]
    return evaluation_measure_id


def get_benchmark_id(cur, benchmark_name):
    cur.execute(
        """
        SELECT id FROM benchmarks
        WHERE name = %s
        """,
        (benchmark_name,),
    )
    benchmark_ids = cur.fetchall()
    if len(benchmark_ids) == 0:
        return

    benchmark_id = benchmark_ids[0]["id"]
    return benchmark_id


def insert_result(
    benchmark_name, architecture_id, evaluation_measure_name, value, dotenv_fp=None
):
    # load parameters from dotenv file
    load_dotenv(dotenv_path=dotenv_fp, override=True)
    # connect to db using loaded parameters
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        sslmode="require",
    )
    # get required ids from names
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    benchmark_id = get_benchmark_id(cur, benchmark_name)
    assert benchmark_id is not None, f"Benchmark {benchmark_name} not found."
    evaluation_measure_id = get_evaluation_measure_id(cur, evaluation_measure_name)
    assert (
        evaluation_measure_id is not None
    ), f"Evaluation measure {evaluation_measure_name} not found."
    # insert new result
    cur.execute(
        """
        INSERT INTO benchmark_result
            (benchmark, network_architecture, evaluation_measure, value)
        VALUES
            (%s, %s, %s, %s)
        RETURNING id
        """,
        (benchmark_id, architecture_id, evaluation_measure_id, value),
    )
    res_id = cur.fetchone()[0]
    conn.commit()
    # close
    cur.close()
    conn.close()
    return res_id

