"""Train, validate, evaluate, and save all models.  Usage:  python train_models.py"""
from src.training import train_and_evaluate

if __name__ == "__main__":
    meta = train_and_evaluate()
    print("Training finished.")
    print("Recommended model:", meta["recommended"])
    print("Saved models to models/ and evidence to evaluation/")
