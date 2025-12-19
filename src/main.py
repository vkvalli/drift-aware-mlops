import sys
import os
import pathlib

import numpy as np
import torch
import mlflow
import codecarbon


def main():
    print("=== Environment sanity check ===")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"Numpy:   {np.__version__}")
    print(f"Torch:   {torch.__version__}")
    print(f"MLflow:  {mlflow.__version__}")
    print(f"CodeCarbon: {codecarbon.__version__}")
    print()

    # Simple tensor computation to confirm torch works
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    dummy_loss = torch.dot(x, y).item()
    print(f"Dummy torch dot product (loss): {dummy_loss}")

    # --- MLflow setup ---
    project_root = pathlib.Path(__file__).resolve().parent.parent
    os.chdir(project_root)  # make sure CWD is the project root

    print(f"\nProject root: {project_root}")
    print(f"Current working dir: {pathlib.Path().resolve()}")
    print("MLflow will use ./mlruns under this directory.\n")

    # Use default local FileStore (./mlruns)
    mlflow.set_experiment("dev_env_sanity_check")

    with mlflow.start_run(run_name="env_check_run") as run:
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("torch_version", torch.__version__)
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_metric("dummy_loss", dummy_loss)

        tracking_dir = project_root / "mlruns"
        print("MLflow run created:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Experiment ID: {run.info.experiment_id}")
        print(f"  Tracking directory: {tracking_dir}")

    print("\nDone. You should now see an 'mlruns/' folder in your project root.")


if __name__ == "__main__":
    main()
