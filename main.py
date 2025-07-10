import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid
from collections import defaultdict

VIDEO_PATH = "15sec_input_720p.mp4"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "output/tracked_frames"
OUTPUT_VIDEO_PATH = "output/final_video_tacticalmap.mp4"

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
            # Update position for existing track
            x1, y1, x2, y2 = map(int, bbox)
            active_tracks[track_id]['last_position'] = ((x1+x2)/2, (y1+y2)/2)
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
    
    # Initialize new track with position data
    active_tracks[track_id] = {
        'global_id': global_id,
        'features': features,
        'last_position': ((x1+x2)/2, (y1+y2)/2) 
    }
    
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




def get_team_assignment(bbox, frame_width):
    """Simple team assignment based on player position"""
    x_center = (bbox[0] + bbox[2]) / 2
    return "Team A" if x_center < frame_width/2 else "Team B"



def init_minimap(frame):
    """Create blank minimap with field markings"""
    h, w = frame.shape[:2]
    minimap = np.zeros((h//5, w//5, 3), dtype=np.uint8)
    
    # Draw field markings (simplified)
    cv2.rectangle(minimap, (0,0), (minimap.shape[1], minimap.shape[0]), (0,100,0), -1)
    cv2.line(minimap, (minimap.shape[1]//2, 0), (minimap.shape[1]//2, minimap.shape[0]), (255,255,255), 1)
    cv2.circle(minimap, (minimap.shape[1]//2, minimap.shape[0]//2), 30, (255,255,255), 1)
    return minimap



def update_minimap(minimap, tracks, frame_shape):
    """Update minimap with current player positions"""
    minimap_copy = minimap.copy()
    for tid, data in tracks.items():
        if 'last_position' in data:
            # Convert frame coordinates to minimap coordinates
            x = int(data['last_position'][0] * minimap.shape[1] / frame_shape[1])
            y = int(data['last_position'][1] * minimap.shape[0] / frame_shape[0])
            
            # Determine color based on team
            if "United" in data.get('team',''):
                color = (0, 0, 255)  # Red for Man Utd
            elif "City" in data.get('team',''):
                color = (255, 0, 0)   # Blue for Man City
            else:
                color = (255, 255, 0) # Yellow for others
            
            cv2.circle(minimap_copy, (x,y), 3, color, -1)
    return minimap_copy


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
    
     # Initialize minimap on first frame
    if not hasattr(draw_frame, 'minimap_template'):
        draw_frame.minimap_template = init_minimap(frame)
    
    # Initialize team names on first frame
    if not hasattr(draw_frame, 'team_names'):
        draw_frame.team_names = ("Team A", "Team B")  # Default names
        if results.boxes.id is not None:
            boxes = results.boxes.data.cpu().numpy()
            draw_frame.team_names = detect_team_sides(frame, boxes)
    
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
            elif label == 'ball':
                text = "Ball"
            else:  # Player or goalkeeper
                x_center = (x1 + x2) / 2
                team = draw_frame.team_names[0] if x_center < frame.shape[1]/2 else draw_frame.team_names[1]
                text = f"{team} #{global_id}"

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    
    retire_lost_tracks(current_frame_track_ids)
    
    # Update minimap with current positions
    minimap = update_minimap(draw_frame.minimap_template, active_tracks, frame.shape)
    
    # Overlay minimap on frame (top-right corner)
    map_h, map_w = minimap.shape[:2]
    annotated[10:10+map_h, frame.shape[1]-10-map_w:frame.shape[1]-10] = minimap
    return annotated





def detect_team_sides(frame, boxes):
    """Enhanced version with sanity checks"""
    if len(boxes) < 4:  # Not enough players to determine sides
        return ("Team A", "Team B")
    
    left_team, right_team = defaultdict(int), defaultdict(int)
    
    for *xyxy, _, _, _ in boxes:
        x_center = (xyxy[0] + xyxy[2]) / 2
        if x_center < frame.shape[1]/2:
            left_team['count'] += 1
            left_team['avg_x'] += x_center
        else:
            right_team['count'] += 1
            right_team['avg_x'] += x_center
    
    # Sanity check - teams must have at least 2 players
    if left_team['count'] < 2 or right_team['count'] < 2:
        return ("Team A", "Team B")
    
    return ("Manchester United", "Manchester City") if left_team['avg_x']/left_team['count'] < frame.shape[1]/2 else ("Manchester City", "Manchester United")





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