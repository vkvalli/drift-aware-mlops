import os
import time
import pathlib
from typing import Tuple

import numpy as np
import mlflow
from prometheus_client import Counter, Histogram, start_http_server


# ---- Prometheus metrics ----
INGESTED_RECORDS = Counter(
    "ingested_records_total",
    "Total number of records ingested",
)

BATCHES_PROCESSED = Counter(
    "ingestion_batches_total",
    "Total number of ingestion batches processed",
)

BATCH_LATENCY = Histogram(
    "ingestion_batch_latency_seconds",
    "Time taken to ingest a single batch in seconds",
)

BATCH_SIZE = Histogram(
    "ingestion_batch_size",
    "Size (number of records) per ingested batch",
)


def generate_fake_batch(batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a fake tabular batch with features + binary labels.

    Features: normal noise
    Labels:   0/1 based on feature sum threshold
    """
    X = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, 10))
    sums = X.sum(axis=1)
    y = (sums > 0).astype(int)
    return X, y


def run_ingestion_demo(num_batches: int = 10, batch_size: int = 128) -> None:
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    print(f"Project root: {project_root}")
    print(f"Working dir:  {pathlib.Path().resolve()}")
    print("MLflow will log to ./mlruns and Prometheus metrics will be on :8000/metrics\n")

    # Start Prometheus metrics HTTP server
    start_http_server(8000)
    print("Prometheus metrics server started on http://localhost:8000/")

    # Configure MLflow (default local ./mlruns)
    mlflow.set_experiment("ingestion_demo")

    with mlflow.start_run(run_name="simple_ingestion") as run:
        print(f"MLflow run_id: {run.info.run_id}")

        for step in range(num_batches):
            start = time.time()

            X, y = generate_fake_batch(batch_size=batch_size)
            duration = time.time() - start

            n_records = len(X)

            # Update Prometheus metrics
            INGESTED_RECORDS.inc(n_records)
            BATCHES_PROCESSED.inc()
            BATCH_LATENCY.observe(duration)
            BATCH_SIZE.observe(n_records)

            # Log a few metrics to MLflow
            mlflow.log_metric("batch_size", n_records, step=step)
            mlflow.log_metric("ingestion_latency_sec", duration, step=step)

            print(
                f"[batch {step+1}/{num_batches}] "
                f"records={n_records}, latency={duration:.4f}s"
            )

            # Sleep a bit so you can see metrics change over time
            time.sleep(1.0)

    print("\nIngestion demo finished.")
    print("Prometheus metrics server is still running on http://localhost:8000/")
    input("Press Enter to stop the server and exit... ")


if __name__ == "__main__":
    run_ingestion_demo()
