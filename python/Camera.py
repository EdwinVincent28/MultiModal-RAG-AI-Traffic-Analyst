import cv2
import numpy as np
from vidgear.gears import CamGear
from ultralytics import YOLO
import time
import json
import redis
import os
from rembg import remove

script_dir = os.path.dirname(os.path.abspath(__file__))

tracker_path = os.path.join(script_dir, "custom_tracker.yaml")

# --- Redis Connection ---
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
r_img = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# --- CONFIGURATION ---
options={"STREAM_RESOLUTION": "720p"}
stream_url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
stream = CamGear(source=stream_url, stream_mode=True, logging=False, **options).start()


model = YOLO("yolov8n-seg.pt") 
tracker_data = {}
BUFFER = 100

def get_heading(p1, p2):
    return round(np.degrees(np.arctan2(p1[1] - p2[1], p2[0] - p1[0])), 1)

def get_side(x, y, w, h):
    if x < BUFFER: return "Left"
    if x > w - BUFFER: return "Right"
    if y < BUFFER: return "Top"
    if y > h - BUFFER: return "Bottom"
    return "Center"

def predict_exit_side(angle):
    if angle is None: 
        return "Center"
    if -45 <= angle <= 45: return "Right"
    elif 45 < angle < 135: return "Top"
    elif angle >= 135 or angle <= -135: return "Left"
    elif -135 < angle < -45: return "Bottom"
    return "Center"

def get_dominant_color(image):
    if image is None or image.size == 0:
        return "Unknown"

    h, w = image.shape[:2]
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    center_crop = image[cy1:cy2, cx1:cx2]
    
    if center_crop.size == 0: 
        center_crop = image

    avg_color = np.mean(center_crop, axis=(0, 1))
    b, g, r = avg_color

    colors = {
        "Black": (30, 30, 30),
        "White": (220, 220, 220),
        "Silver": (150, 150, 150),
        "Red": (40, 40, 200),
        "Blue": (200, 40, 40),
        "Green": (40, 200, 40),
        "Yellow": (40, 200, 200)
    }

    best_color = "Unknown"
    min_dist = float('inf')

    for name, (cb, cg, cr) in colors.items():
        dist = (b - cb)**2 + (g - cg)**2 + (r - cr)**2
        if dist < min_dist:
            min_dist = dist
            best_color = name

    return best_color

frame_count = 0 

try:
    while True:
        frame = stream.read()
        if frame is None:
            print("[INFO] Stream lost. Reconnecting in 5 seconds...")
            stream.stop()       # Stop the dead stream
            time.sleep(5)       # Wait before retrying to prevent spamming
            stream = CamGear(source=stream_url, stream_mode=True, logging=False, **options).start()
            continue            # Skip the rest of the loop and try reading again
        # Processes every other frame (halves processing load).
        frame_count += 1
        if frame_count % 4 != 0:
            continue

        
        h, w = frame.shape[:2]
        current_time = time.time()

        results = model.track(
            frame,
            persist=True,
            #tracker="bytetrack.yaml", 
            tracker=tracker_path,
            verbose=False,
            conf=0.15,
            iou=0.4,
            classes=[2, 3, 5, 7]
        )
        
        annotated_frame = results[0].plot(conf=False)
        display_frame = cv2.resize(annotated_frame, (0, 0), fx=0.5, fy=0.5)
        #cv2.imshow("Live Traffic Stream", display_frame)
        
        success, buffer = cv2.imencode('.jpg', display_frame)
        if success:
            r_img.set("traffic:frame:live", buffer.tobytes())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        boxes = results[0].boxes
        
        if boxes.id is not None:
            active_ids = boxes.id.cpu().tolist()
            
            for vid, box, cls_idx in zip(active_ids, boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist()):
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2

                x1, y1, x2, y2 = map(int, box)
                crop_y1, crop_y2 = max(0, y1-25), min(h, y2+25)
                crop_x1, crop_x2 = max(0, x1-25), min(w, x2+25)
                vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if vid not in tracker_data:
                    tracker_data[vid] = {
                        'cls': model.names[int(cls_idx)], 
                        'ent_angle': None,
                        'ent_side': get_side(cx, cy, w, h), 
                        'path': [(cx, cy)], 
                        'ent_time': time.strftime("%H:%M:%S"),
                        'last_seen_time': time.strftime("%H:%M:%S"),
                        'missing_frames': 0,  # Initialize grace period timer
                        'last_crop': vehicle_crop,
                        'best_crop': vehicle_crop,
                        'best_area': (x2 - x1) * (y2 - y1)
                    }
                else:
                    tracker_data[vid]['path'].append((cx, cy))
                    tracker_data[vid]['missing_frames'] = 0
                    tracker_data[vid]['last_seen_time'] = time.strftime("%H:%M:%S")
                    tracker_data[vid]['last_crop'] = vehicle_crop

                    current_area = (x2 - x1) * (y2 - y1)

                    if current_area > tracker_data[vid]['best_area']:
                        tracker_data[vid]['best_area'] = current_area
                        tracker_data[vid]['best_crop'] = vehicle_crop

                    
                    if tracker_data[vid]['ent_angle'] is None and len(tracker_data[vid]['path']) > 3:
                        dist_sq = (cx - tracker_data[vid]['path'][0][0])**2 + (cy - tracker_data[vid]['path'][0][1])**2
                        if dist_sq > 100: #(10 pixels of movement)
                            tracker_data[vid]['ent_angle'] = get_heading(tracker_data[vid]['path'][0], (cx, cy))
        else:
            active_ids = [] # Screen is empty!

        # --- Grace Period & Exit Logic ---
        for vid in list(tracker_data.keys()):
            if vid not in active_ids:
                tracker_data[vid]['missing_frames'] += 1
                
                # Increased to 60 frames (approx 2 seconds) to outlast ByteTrack's internal memory
                if tracker_data[vid]['missing_frames'] > 15:
                    data = tracker_data[vid]
                    path_len = len(data['path'])
                    has_angle = data['ent_angle'] is not None
                    last_p = data['path'][-1]

                    if has_angle and path_len > 5:
                        ext_angle = get_heading(data['path'][-min(7, path_len-1)], last_p) 
                        exit_side = predict_exit_side(ext_angle) # Predict where it went!
                        exit_time = time.strftime('%H:%M:%S')
                        current_sec = int(current_time)
                        
                        latest_frame_key = f"traffic:frame:{current_sec}_vid{vid}"
                        color_label = "Unknown"

                        if data['best_crop'] is not None and data['best_crop'].size > 0:
                            color_label = get_dominant_color(data['best_crop'])

                            bg_removed = remove(
                                data['best_crop'], 
                                alpha_matting=True, 
                                alpha_matting_foreground_threshold=240,
                                alpha_matting_background_threshold=10,
                                alpha_matting_erode_size=5
                            )

                            black_bg = np.zeros_like(data['best_crop'])
                            alpha = bg_removed[:, :, 3] / 255.0
                            for c in range(3):
                                black_bg[:, :, c] = (alpha * bg_removed[:, :, c] + 
                                                    (1 - alpha) * black_bg[:, :, c])
                            
                            success, buffer = cv2.imencode('.jpg', black_bg)
                            if success:
                                frame_bytes = buffer.tobytes()
                                r_img.setex(latest_frame_key, 3600, frame_bytes)

                        event = {
                            "vehicle_id": int(vid),
                            "class": data['cls'],
                            "color": color_label,
                            "entry_side": data['ent_side'],                  
                            "entry_angle": data['ent_angle'],
                            "entry_time": data['ent_time'],
                            "exit_side": exit_side, 
                            "exit_angle": ext_angle,
                            "exit_time": exit_time,
                            "timestamp": current_time,
                            "linked_frame": latest_frame_key                 
                        }

                        try:
                            pipe = r.pipeline()
                            pipe.lpush("traffic:events", json.dumps(event))
                            pipe.ltrim("traffic:events", 0, 999) 
                            pipe.hset(f"traffic:latest:{data['cls']}", mapping={
                                "entry_angle": data['ent_angle'],
                                "exit_angle": ext_angle,
                                "exit_time": exit_time
                            })
                            pipe.incr(f"traffic:count:{data['cls']}")
                            pipe.execute()
                            print(f"[REDIS ✓] SAVED: ID: {vid} | Cat: {data['cls']} | Ent: {data['ent_side']} ({data['ent_angle']}°) | Ext: {exit_side} ({ext_angle}°) | Color: {color_label}")
                        except redis.exceptions.ConnectionError:
                            print("[ERROR] Could not connect to Redis!")

                    # The grace period has completely expired, safe to delete
                    del tracker_data[vid]

except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    cv2.destroyAllWindows()