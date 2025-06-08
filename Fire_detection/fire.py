from ultralytics import YOLO


# Load the YOLO model
model = YOLO("yolo11n.pt")

# Train the model with custom dataset
model.train(data="B:/PAD_PROJECT/Data_fire/data_fire.yaml", epochs=50, imgsz=640, batch=16, device="cuda")  

# Save the model
model.save("yolo11n_fire.pt")

 