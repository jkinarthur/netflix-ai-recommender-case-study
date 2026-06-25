from pathlib import Path

from src.recommender_pipeline import run_pipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    metrics = run_pipeline(root, k=10)
    print("\n=== Model Comparison (Top-K=10) ===")
    print(metrics.to_string(index=False))
    print("\nOutputs written to:", root / "outputs")
