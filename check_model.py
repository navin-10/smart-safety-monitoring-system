import os
from ultralytics import YOLO

# 1. Define model path
model_path = os.path.join("models", "best.pt")

# 2. Check if the file exists
if not os.path.exists(model_path):
    print(f"[ERROR] 'best.pt' not found at: {model_path}")
    print("Please make sure best.pt is placed inside the 'models' folder.")
else:
    # 3. Load the model
    print("[INFO] Loading your trained model...")
    model = YOLO(model_path)
    
    # 4. Print the detected classes
    print("\n--- MODEL CLASSES DETECTED ---")
    for class_id, class_name in model.names.items():
        print(f"ID {class_id}: {class_name}")
    print("------------------------------\n")
    print("[SUCCESS] Your model is ready for real-time inference!")