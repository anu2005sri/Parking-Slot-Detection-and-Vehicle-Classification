import cv2
import pytesseract
import re
import pickle
import cvzone
import numpy as np
import pyttsx3
import threading
import time
import os

# ===================== Config =====================
# Tesseract path (Windows) - update if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Haar Cascade for number plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Parking box size
width, height = 107, 48

# Column grouping threshold
COLUMN_THRESHOLD = int(width * 0.6)

# Plate regex and log file
plate_pattern = re.compile(r'[A-Z0-9\s]+')
output_file = "detected_vehicle_info.txt"
screenshot_folder = "plate_screenshots"

# Make folder for plate images if not exists
os.makedirs(screenshot_folder, exist_ok=True)
# ==================================================

# TTS
engine = pyttsx3.init()
def speak_instruction(msg):
    engine.say(msg)
    engine.runAndWait()

def save_to_file(text):
    with open(output_file, 'a') as f:
        f.write(text + '\n')

# Load parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# ---- Build vertical order ----
def build_vertical_order(positions, col_thresh):
    xs_sorted = sorted(positions, key=lambda p: p[0])
    columns = []
    for p in xs_sorted:
        x, y = p
        placed = False
        for col in columns:
            if abs(x - col['x_ref']) <= col_thresh:
                col['spots'].append(p)
                col['xs'].append(x)
                col['x_ref'] = int(np.median(col['xs']))
                placed = True
                break
        if not placed:
            columns.append({'x_ref': x, 'xs': [x], 'spots': [p]})

    columns.sort(key=lambda c: c['x_ref'])

    ordered = []
    for col in columns:
        col['spots'].sort(key=lambda p: p[1])
        ordered.extend(col['spots'])

    pos_to_idx = {pos: i for i, pos in enumerate(ordered)}
    return ordered, pos_to_idx

ordered_positions, pos_to_idx = build_vertical_order(posList, COLUMN_THRESHOLD)

# Global trackers
car_in_time = {}
car_out_time = {}
prev_free_count = None
last_detected_plate = None  # Keep track of last detected plate

# ===================== Plate Detection =====================
def detect_and_recognize_plate_from_webcam():
    global last_detected_plate
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 30))

        for (x, y, w, h) in plates:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]  # colored ROI for saving screenshot

            # OCR
            text = pytesseract.image_to_string(roi_gray, config='--psm 8')
            matches = plate_pattern.findall(text)

            for match in matches:
                # ✅ Clean plate text: remove spaces & non-alphanumeric
                plate_clean = re.sub(r'[^A-Z0-9]', '', match.strip())
                if len(plate_clean) < 5:  # ignore junk OCR
                    continue

                last_detected_plate = plate_clean
                detected_text = f"Detected Number Plate: {last_detected_plate}"
                print(detected_text)
                save_to_file(detected_text)

                # ✅ Save screenshot of number plate
                timestamp = int(time.time())
                filename = os.path.join(screenshot_folder, f"{last_detected_plate}_{timestamp}.png")
                cv2.imwrite(filename, roi_color)
                print(f"Saved screenshot: {filename}")

                # Trigger parking check once plate detected
                check_parking_space()

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Plate", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== Parking Check =====================
def check_parking_space():
    global prev_free_count, last_detected_plate
    cap = cv2.VideoCapture('carPark.mp4')

    free_slots = []

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(
            imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 16
        )
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        spaceCounter = 0
        free_slots.clear()

        for idx, pos in enumerate(ordered_positions):
            x, y = pos
            imgCrop = imgDilate[y:y + height, x:x + width]
            count = cv2.countNonZero(imgCrop)

            occupied = count >= 900

            if occupied:
                color = (0, 0, 255)
                thickness = 2
                if idx not in car_in_time:
                    car_in_time[idx] = time.time()
            else:
                color = (0, 255, 0)
                thickness = 5
                spaceCounter += 1
                free_slots.append(idx+1)
                if idx in car_in_time:
                    stay = time.time() - car_in_time[idx]
                    car_out_time[idx] = stay
                    mm, ss = divmod(int(stay), 60)
                    msg = f"Car left Box {idx+1} after {mm:02}:{ss:02}"
                    print(msg)
                    save_to_file(msg)
                    del car_in_time[idx]

            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

            if idx in car_in_time:
                dur = time.time() - car_in_time[idx]
                mm, ss = divmod(int(dur), 60)
                label = f"{idx+1} {mm:02}:{ss:02}"
            else:
                label = f"{idx+1}"

            cvzone.putTextRect(img, label, (x, y + 18), scale=1,
                               thickness=2, offset=4, colorR=color)

        # Free counter
        cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(ordered_positions)}',
                           (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

        # ✅ Audio: announce plate + slot
        if last_detected_plate and spaceCounter > 0:
            slot_assigned = free_slots[0]
            instruction = f"There are {spaceCounter} free parking spaces available. {last_detected_plate} number car go to slot number {slot_assigned}."
            print(instruction)
            save_to_file(instruction)
            threading.Thread(target=speak_instruction, args=(instruction,)).start()
            last_detected_plate = None

        elif last_detected_plate and spaceCounter == 0:
            instruction = f"No parking spaces available for {last_detected_plate} car, please wait."
            print(instruction)
            save_to_file(instruction)
            threading.Thread(target=speak_instruction, args=(instruction,)).start()
            last_detected_plate = None

        cv2.imshow("Parking Lot", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Start -----------------
detect_and_recognize_plate_from_webcam()
