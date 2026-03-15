from __future__ import annotations

from src.data_processing import save_raw_and_processed_data
from src.evaluate_model import evaluate
from src.train_model import train_and_select_best


def run_pipeline() -> None:
    save_raw_and_processed_data()
    train_and_select_best()
    evaluate()


if __name__ == "__main__":
    run_pipeline()
    print("End-to-end churn pipeline complete.")
