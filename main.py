import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid



VIDEO_PATH = "/Users/sheenamittal/Desktop/work /My Projects/Internship_assignment/15sec_input_720p.mp4"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "/Users/sheenamittal/Desktop/work /My Projects/Internship_assignment/output/tracked_frames"
OUTPUT_VIDEO_PATH = "/Users/sheenamittal/Desktop/work /My Projects/Internship_assignment/output/final_video_with_jersey_2.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

global_id_counter = 0
active_tracks = {}
inactive_gallery = []

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load re-identification model
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.to(device)
reid_model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

JERSEY_COLOR_TO_TEAM = {
    "white": "Real Madrid",
    "blue": "Chelsea",
    "red": "Manchester United",
    "yellow": "Brazil",
    "skyblue": "Argentina",
    "darkblue": "France",
    "green": "Mexico",
    "black": "New Zealand",
    "orange": "Wolverhampton",
    "purple": "Fiorentina",
    "pink": "Palermo",
    "maroon": "West Ham United",
    "lightblue": "Manchester City",
    "gold": "LA Galaxy",
    "grey": "Juventus (away)",
    # Add more as needed
}


def extract_features(image_crop):
    try:
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = reid_model(tensor)
        return features.cpu().numpy().flatten()
    except:
        return None

def match_in_gallery(features, used_global_ids, threshold=0.7):
    filtered = [g for g in inactive_gallery if g['global_id'] not in used_global_ids]
    if not filtered:
        return None
    gallery_features = [g['features'] for g in filtered]
    gallery_ids = [g['global_id'] for g in filtered]
    sims = cosine_similarity([features], gallery_features)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] > threshold:
        return gallery_ids[best_idx]
    return None

def assign_global_id(track_id, bbox, frame, used_global_ids):
    global global_id_counter
    if track_id in active_tracks:
        global_id = active_tracks[track_id]['global_id']
        if global_id not in used_global_ids:
            return global_id
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    features = extract_features(crop)
    if features is None:
        return None
    matched_global_id = match_in_gallery(features, used_global_ids)
    if matched_global_id is not None:
        global_id = matched_global_id
    else:
        global_id_counter += 1
        global_id = global_id_counter
        inactive_gallery.append({'global_id': global_id, 'features': features})
    active_tracks[track_id] = {'global_id': global_id, 'features': features}
    used_global_ids.add(global_id)
    return global_id

def retire_lost_tracks(current_track_ids):
    lost_ids = set(active_tracks.keys()) - set(current_track_ids)
    for tid in lost_ids:
        if 'features' in active_tracks[tid]:
            inactive_gallery.append({
                'global_id': active_tracks[tid]['global_id'],
                'features': active_tracks[tid]['features']
            })
    for tid in lost_ids:
        del active_tracks[tid]

def draw_frame(frame, results, class_names):
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_map = {
        'player': (255, 255, 255),
        'referee': (0, 215, 255),
        'goalkeeper': (0, 0, 255),
        'ball': (0, 255, 0)
    }
    used_global_ids = set()
    current_frame_track_ids = []
    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            if label not in ['player', 'referee', 'goalkeeper', 'ball']:
                continue
            current_frame_track_ids.append(track_id)
            global_id = assign_global_id(track_id, (x1, y1, x2, y2), frame, used_global_ids)
            color = color_map.get(label, (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            if label == 'referee':
                text = "Referee"
            elif label in ['player', 'goalkeeper']:
                # Only do jersey detection for players and goalkeepers
                jersey_crop = frame[y1:y1 + (y2 - y1) // 2, x1:x2]  # upper half of the bounding box
                team_name = detect_team_name(jersey_crop)
                text = f"{team_name} #{global_id}"
            elif label == 'ball':
                text = "Ball"
            else:
                text = label

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    retire_lost_tracks(current_frame_track_ids)
    return annotated

def detect_team_name(crop):
    # Focus on the central region of the crop
    h, w, _ = crop.shape
    crop = crop[int(0.1*h):int(0.6*h), int(0.2*w):int(0.8*w)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv.reshape((-1, 3))

    # Use 3 clusters for better separation
    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(hsv_pixels)
    centers = kmeans.cluster_centers_

    # Pick the cluster with the highest saturation and value (likely the jersey)
    idx = np.argmax(centers[:,1] * centers[:,2])
    h, s, v = centers[idx]

    print(f"Dominant HSV: {h:.1f}, {s:.1f}, {v:.1f}")  # Debugging


    # Example: Add more teams with their HSV ranges
    if s < 40 and v > 200:
        return "Real Madrid"
    elif 0 <= h <= 10 or 160 <= h <= 180:
        return "Manchester United"
    elif 15 <= h <= 25 and s > 100 and v > 100:
        return "Wolverhampton"  # Example: orange
    elif 100 <= h <= 130:
        return "Chelsea"
    elif 35 <= h <= 85 and s > 100:
        return "Norwich City"  # Example: greenish-yellow
    elif 20 <= h <= 40:
        return "Brazil"
    elif 90 <= h <= 100:
        return "Argentina"
    elif 110 <= h <= 130:
        return "France"
    elif 50 <= h <= 80:
        return "Mexico"
    elif 120 <= h <= 140 and s < 80:
        return "Tottenham Hotspur"  # Example: light blue/white
    elif v < 50:
        return "New Zealand"
    # Add more teams as needed
    else:
        return "Unknown"




def run_tracking(video_path, model_path, output_dir, output_video_path):
    model = YOLO(model_path)
    class_names = model.names
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(source=frame, persist=True, conf=0.4, verbose=False)[0]
        output_frame = draw_frame(frame, results, class_names)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), output_frame)
        out_writer.write(output_frame)
        frame_idx += 1
    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Video saved to {output_video_path}")

if __name__ == "__main__":
    run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)
