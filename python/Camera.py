import cv2
import numpy as np
from vidgear.gears import CamGear
from ultralytics import YOLO
import time
import json
import redis
import os

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

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    v_channel = hsv_img[:, :, 2]
    valid_mask = v_channel > 15
    
    valid_pixels = hsv_img[valid_mask]
    
    if len(valid_pixels) == 0:
        return "Black"
        
    h = valid_pixels[:, 0]
    s = valid_pixels[:, 1]
    v = valid_pixels[:, 2]
    
    # Catch pixels that are technically blue, but are too dark/desaturated to be real paint.
    is_fake_blue = (h >= 85) & (h < 140) & (s < 90) & (v < 110)
    
    # Give the Black bucket a slightly higher base threshold, and feed it the Fake Blue votes!
    is_black = (v < 55) | is_fake_blue
    
    is_white = (s < 45) & (v > 140) & ~is_black
    is_silver = (s < 45) & (v <= 140) & ~is_black
    
    is_colored = (s >= 45) & ~is_black
    
    is_red = is_colored & ((h < 10) | (h > 165))
    is_yellow = is_colored & (h >= 10) & (h < 35)
    is_green = is_colored & (h >= 35) & (h < 85)
    is_blue = is_colored & (h >= 85) & (h < 140) # Only REAL blue paint gets to vote here
    
    # 3. COUNT THE VOTES
    counts = {
        "Black": np.sum(is_black),
        "White": np.sum(is_white),
        "Silver": np.sum(is_silver),
        "Red": np.sum(is_red),
        "Yellow": np.sum(is_yellow),
        "Green": np.sum(is_green),
        "Blue": np.sum(is_blue)
    }
    
    # 4. DECIDE THE WINNER
    total_pixels = len(valid_pixels)
    color_votes = counts["Red"] + counts["Yellow"] + counts["Green"] + counts["Blue"]
    
    # THE 15% RULE:
    # If at least 15% of the car's body is a vibrant color, it is a colored car.
    if total_pixels > 0 and (color_votes / total_pixels) > 0.15:
        # Find the color with the most votes
        color_candidates = {k: counts[k] for k in ["Red", "Yellow", "Green", "Blue"]}
        return max(color_candidates, key=color_candidates.get)
    else:
        # If there is no vibrant color, it is a Grayscale car
        grayscale_candidates = {k: counts[k] for k in ["Black", "White", "Silver"]}
        return max(grayscale_candidates, key=grayscale_candidates.get)

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

        
        h_frame, w_frame = frame.shape[:2]
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
        masks = results[0].masks
        
        if boxes.id is not None:
            active_ids = boxes.id.cpu().tolist()
            polygons = masks.xy if masks is not None else []
            
            for idx, (vid, box, cls_idx) in enumerate(zip(active_ids, boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist())):
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                x1, y1, x2, y2 = map(int, box)
                
                # Small padding is fine now because the mask is highly accurate
                crop_y1, crop_y2 = max(0, y1-5), min(h_frame, y2+5)
                crop_x1, crop_x2 = max(0, x1-5), min(w_frame, x2+5)

                # --- NEW: APPLY SEGMENTATION MASK TO CUT OUT VEHICLE ---
                poly = polygons[idx] if idx < len(polygons) else None
                
                if poly is not None and len(poly) > 0:
                    # 1. Create empty black canvas
                    frame_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    # 2. Draw the exact polygon outline in white
                    cv2.fillPoly(frame_mask, [np.array(poly, dtype=np.int32)], 255)
                    # 3. Punch out the car (turns everything else pitch black)
                    masked_frame = cv2.bitwise_and(frame, frame, mask=frame_mask)
                    # 4. Crop to the box size
                    vehicle_crop = masked_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                else:
                    vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if vid not in tracker_data:
                    tracker_data[vid] = {
                        'cls': model.names[int(cls_idx)], 
                        'ent_angle': None,
                        'ent_side': get_side(cx, cy, w_frame, h_frame), 
                        'path': [(cx, cy)], 
                        'ent_time': time.strftime("%H:%M:%S"),
                        'last_seen_time': time.strftime("%H:%M:%S"),
                        'missing_frames': 0,
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
                        if dist_sq > 100: 
                            tracker_data[vid]['ent_angle'] = get_heading(tracker_data[vid]['path'][0], (cx, cy))
        else:
            active_ids = []

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

                            success, buffer = cv2.imencode('.jpg', data['best_crop'])
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