from ultralytics import YOLO

def train_model():
    # Load the YOLO model
    model = YOLO("yolo11n.pt")

    # Train the model with a custom dataset
    model.train(
        data="B:/PAD_PROJECT/Data_fire/data_fire.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda"  # Ensure it runs on GPU
    )

    # Save the model
    model.save("yolo11n_fire.pt")

if __name__ == "__main__":  # <-- Add this safeguard
    train_model()
