from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict(
    source="valid/images/palet47_PNG.rf.a8f1119859b4c567bc65f3278ff72c1c.jpg",  # bisa ganti path gambar/video
    conf=0.3,
    save=True
)