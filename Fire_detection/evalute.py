from ultralytics import YOLO

def main():
    # Load the YOLO model
    model = YOLO("yolo11n.pt")

    # Validate the model with custom dataset
    metrics = model.val(data="B:/PAD_PROJECT/Data_fire/data_fire.yaml")

    # Extract metrics
    print("\n🔹 Model Evaluation Results 🔹")
    print(f"📌 mAP@50: {metrics.box.map:.4f}")  # Mean Average Precision at IoU 0.50
    print(f"📌 mAP@50-95: {metrics.box.map50_95:.4f}")  # Mean Average Precision at different IoUs
    print(f"📌 Precision: {metrics.box.p:.4f}")
    print(f"📌 Recall: {metrics.box.r:.4f}")

if __name__ == "__main__":
    main()  # Ensures the script runs properly in multiprocessing environments

# Training loss can be found in results.csv inside runs/detect/train/
import pandas as pd
import matplotlib.pyplot as plt

# Load loss data
df = pd.read_csv("C:/Users/byred/runs/detect/train14/results.csv")

# Plot losses
plt.plot(df["train/box_loss"], label="Train Box Loss")
plt.plot(df["val/box_loss"], label="Val Box Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
