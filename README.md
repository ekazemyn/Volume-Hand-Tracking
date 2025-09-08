# Hand Volume Control

Control your computer's volume using hand gestures! Move your thumb and index finger closer or farther apart to adjust the volume.

## What it does

- Detects your hand through your webcam
- Measures distance between thumb and index finger
- Changes system volume based on finger distance
- Shows volume bar and hand tracking on screen

## Requirements

- Windows computer
- Webcam
- Python 3.7+

## Installation

1. Install Python packages:
```bash
pip install opencv-python mediapipe numpy pycaw comtypes
```

2. Run the program:
```bash
python hand_volume_control.py
```

## How to use

1. Put your hand in front of the camera
2. Bring thumb and index finger close together = lower volume
3. Move thumb and index finger apart = higher volume
4. Press 'q' to quit


That's it! Enjoy controlling your volume with hand gestures.
