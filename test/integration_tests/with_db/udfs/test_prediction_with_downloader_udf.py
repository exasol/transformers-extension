import time
from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality,
)
from test.utils import postprocessing

TASK_TYPE = "filling_mask"
SUB_DIR = "test_downloader_with_prediction_sub_dir"
MODEL_NAME = "gaunernst/bert-tiny-uncased"


def test_prediction_with_downloader_udf(setup_database, db_conn, bucketfs_location):
    bucketfs_conn_name, _ = setup_database

    try:
        # execute downloader UDF
        input_data = (MODEL_NAME, TASK_TYPE, SUB_DIR, bucketfs_conn_name, "")
        query = f"""
            SELECT TE_MODEL_DOWNLOADER_UDF(
            t.model_name,
            t.task_type,
            t.sub_dir,
            t.bucketfs_conn_name,
            t.token_conn_name
            ) FROM (VALUES {str(input_data)} AS
            t(model_name, task_type, sub_dir, bucketfs_conn_name, token_conn_name));
            """

        result = db_conn.execute(query).fetchall()
        time.sleep(10)

        # execute the filling mask UDF
        text_data = "I <mask> you so much."
        top_k = 3
        input_data = ("", bucketfs_conn_name, SUB_DIR, MODEL_NAME, text_data, top_k)

        query = (
            f"SELECT TE_FILLING_MASK_UDF("
            f"t.device_id, "
            f"t.bucketfs_conn_name, "
            f"t.sub_dir, "
            f"t.model_name, "
            f"t.text_data,"
            f"t.top_k"
            f") FROM (VALUES {str(input_data)} "
            f"AS t(device_id, bucketfs_conn_name, sub_dir, "
            f"model_name, text_data, top_k));"
        )

        result = db_conn.execute(query).fetchall()

        # assertions
        assert len(result) == top_k
        assert all(row[-1] is None for row in result)

        acceptable_results = ["love", "miss", "want", "need"]
        assert_lenient_check_of_output_quality(result, acceptable_results, 0.5, 5)

    finally:
        postprocessing.cleanup_buckets(bucketfs_location, SUB_DIR)
