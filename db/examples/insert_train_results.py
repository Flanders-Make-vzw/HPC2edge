# %% imports
import traceback
import psycopg2
from dotenv import load_dotenv
import os
import psycopg2.extras


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
        sslmode="verify-full",
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


# %% example usage
if __name__ == "__main__":
    # get this when inserting in the `network_architectures` table (cf. lines 71 to 75)
    current_architecture_id = 1
    # computed loss
    loss = 0.1234
    try:
        res_id = insert_result(
            "RAISE-LPBF-Laser training hold-out",
            current_architecture_id,
            "root_mean_squared_error",
            loss,
            dotenv_fp="_env",
        )
    except Exception:
        print("Inserting new result in DB failed:" + traceback.format_exc())
    # Rest of the code here
