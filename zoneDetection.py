import cv2
import json
import os
import csv
import requests  # pastikan sudah install: pip install requests
from ultralytics import YOLO
from datetime import datetime

# === NX BOOKMARK INTEGRATION STEPS ===
# 1. Dapatkan NX API URL, cameraId/UUID, username, password
# 2. Pastikan 'requests' sudah terinstall (pip install requests)
# 3. Fungsi create_nx_bookmark akan dipanggil saat pallet terdeteksi
# 4. Bookmark akan muncul di timeline kamera di NX Client
# === NX API CONFIG ===
NX_CAMERA_ID = "3c2a68b1-a310-a52f-1c33-e1c7e5de0eea"  # Ganti dengan cameraId/UUID dari NX
NX_API_URL = f"https://192.168.2.226:7001/rest/v3/devices/{NX_CAMERA_ID}/bookmarks"
NX_AUTH = ('admin', 'rangga7671234')  # Ganti dengan kredensial NX Anda

def create_nx_bookmark(camera_id, start_time, end_time, description):
    duration = end_time - start_time
    payload = {
        "name": description,
        "description": "",
        "startTimeMs": start_time,
        "durationMs": duration,
        "tags": [""]
    }
    print(f"[NX] Sending bookmark payload: {payload}")
    try:
        response = requests.post(
            NX_API_URL, json=payload, auth=NX_AUTH, verify=False
        )
        print(f"[NX] Response status: {response.status_code}")
        print(f"[NX] Response text: {response.text}")
        if response.status_code == 200:
            print(f"[NX] Bookmark berhasil dikirim: {description}")
        else:
            print(f"[NX] Gagal kirim bookmark: {response.text}")
    except Exception as e:
        print(f"[NX] Error: {e}")

# === 1. Setup CSV ===
csv_file = "pallet_count.csv"

# Cek apakah file ada, kalau tidak buat dan tambahkan header
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Pallet ID", "Count"])
    print(f"[INFO] CSV '{csv_file}' dibuat dengan header.")

# === 2. Load YOLO model ===
model = YOLO("runs/detect/train/weights/model-v2.pt")

# === 3. Buka video / webcam ===
cap = cv2.VideoCapture("sample-vid.mp4")  # atau rtsp://...
# cap = cv2.VideoCapture(
#     "rtsp://admin:rangga7671234@192.168.2.226:7001/3c2a68b1-a310-a52f-1c33-e1c7e5de0eea?stream=0"
# )
# === 4. Zone management ===
zone_file = "zone.json"
zone_start, zone_end = None, None
drawing = False

# Load zona kalau file ada
if os.path.exists(zone_file):
    with open(zone_file, "r") as f:
        data = json.load(f)
        zone_start = tuple(data["zone_start"])
        zone_end = tuple(data["zone_end"])
        print(f"[INFO] Zona dimuat dari {zone_file}: {zone_start} -> {zone_end}")

def draw_zone(event, x, y, flags, param):
    global zone_start, zone_end, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        zone_start = (x, y)
        zone_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        zone_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zone_end = (x, y)
        # Simpan ke file JSON
        with open(zone_file, "w") as f:
            json.dump({"zone_start": zone_start, "zone_end": zone_end}, f)
        print(f"[INFO] Zona disimpan ke {zone_file}: {zone_start} -> {zone_end}")

cv2.namedWindow("Pallet Detection")
cv2.setMouseCallback("Pallet Detection", draw_zone)

output_file = "output_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (800, 600))  # 20 fps, ukuran sesuai resize


# === 5. Counting Pallet ===
pallet_count = 0
already_counted = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5)

    if results[0].boxes.id is not None and zone_start and zone_end:
        for box, track_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(track_id)
            confidence = float(conf)

            # bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"Pallet {obj_id}, {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # center object
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # check masuk zona
            if (zone_start[0] < cx < zone_end[0] and
                zone_start[1] < cy < zone_end[1]):
                if obj_id not in already_counted:
                    pallet_count += 1
                    already_counted.add(obj_id)

                    # update CSV
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([now, f"Pallet-{obj_id}", pallet_count])
                    print(f"[INFO] Pallet-{obj_id} counted. Total: {pallet_count}")

                    # === KIRIM BOOKMARK KE NX ===
                    # Waktu bookmark: gunakan waktu sekarang (epoch ms)
                    now_epoch = int(datetime.now().timestamp() * 1000)
                    create_nx_bookmark(
                        NX_CAMERA_ID,
                        now_epoch,  # startTime
                        now_epoch + 5000,  # endTime (5 detik)
                        f"Pallet-{obj_id} counted: {pallet_count}"
                    )

    # gambar zona
    if zone_start and zone_end:
        # Buat overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, zone_start, zone_end, (0,0,255), -1)  # isi merah
        alpha = 0.2  # transparansi
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        # Garis tepi zona
        cv2.rectangle(frame, zone_start, zone_end, (0,0,255), 2)
    frame_resized = cv2.resize(frame, (800, 600))  # width=800, height=600
    cv2.putText(frame_resized, f"Count: {pallet_count}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    # Simpan frame ke video output
    out.write(frame_resized)

    # cv2.putText(frame_resized, f"Count: {pallet_count}", (50,50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Pallet Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  
cv2.destroyAllWindows()