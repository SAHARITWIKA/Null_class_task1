import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataset_loader import load_imdb_dataset
from model_builder import build_age_model

MAT_FILE = "./data/imdb_crop/imdb.mat"
DATASET_PATH = "./data/imdb_crop"

MODEL_SAVE_PATH = "../models/vggface_age_model.h5"

def main():
    print("[INFO] Loading dataset...")
    X, y = load_imdb_dataset(MAT_FILE, DATASET_PATH, max_images=30000)

    print(f"[INFO] Dataset loaded. Total samples: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Building model...")
    model = build_age_model()

    print("[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32
    )

    print("[INFO] Saving model...")
    model.save(MODEL_SAVE_PATH)

    print("[INFO] Training complete. Plotting loss...")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
