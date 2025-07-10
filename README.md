
# Football Player Tracking and Tactical Map with Re-Identification

This project performs **real-time player tracking, re-identification, team classification, and tactical minimap visualization** using **YOLOv8** and **Torchreid**.




## Demo

![Demo](file:///Users/sheenamittal/Desktop/work%20/My%20Projects/Football%20player%20tracking%20system/output/tracked_frames/frame_0012.jpg)



##  Objective

Build a pipeline that:
1. Detects players, referees, goalkeepers, and the ball.
2. Assigns **global player IDs** using deep re-identification.
3. Classifies players into **Manchester United (red)** or **Manchester City (blue)**.
4. Displays a **live minimap** showing tactical positions of all players.

---


## Core Concept

1. **YOLOv8 Model**
Detects objects: player, goalkeeper, referee, ball

Assigns local track_id to each object

2. **Torchreid Model (osnet_x1_0)**
Extracts robust deep features for cropped players

Matches new players to a gallery of previously seen players

Ensures persistent global player identity (global_id)

3. **Team Detection Logic**
Assigns teams based on initial player distribution

Left half → "Manchester United"

Right half → "Manchester City"

4. **Minimap (Tactical Map)**
Bird’s eye view of the pitch

Tracks live position of each player by global ID and team

Red = Manchester United, Blue = Manchester City



## Project Structure
``` bash
.
├── main.py                  # Main tracking pipeline
├── best.pt                 # Trained YOLOv8 detection model
├── 15sec_input_720p.mp4    # Sample input video
├── output/
│   ├── tracked_frames/     # Saved annotated frames
│   └── final_video_tacticalmap.mp4
├── README.md

```


## Requirements
``` bash
pip install -r requirements.txt

```


