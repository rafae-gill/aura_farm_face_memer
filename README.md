# Aura Farm Meme Face FR FR 👶🐵

Real-time face and hand-gesture tracking that mirrors your expressions as memes: pull a face
at your webcam and a baby, a monkey or Shrek pulls it back. Built for a friend, for fun.

## How it works

Each webcam frame goes through two [MediaPipe](https://ai.google.dev/edge/mediapipe) models:

- **Face Landmarker**: 478 face landmarks plus *blendshapes*, scores for things like "mouth
  open" or "tongue out". Head pose (turned, tilted, looking down) is estimated from the
  landmarks.
- **Hand Landmarker**: 21 points per hand, used to tell where your hands are relative to
  your face.

A small set of rules maps that to a gesture, and the gesture to a meme. Both models are
downloaded automatically on first run.

## Gestures

| Do this | You get |
|---|---|
| Normal face | 👶 Baby |
| Fist near your face | 👶 Baby with hand raised |
| Turn your head + open your mouth | 👶 Baby looking to the side |
| Tilt down + stick your tongue out | 👶 Baby looking down |
| Extended finger near your face | 🐵 Monkey biting finger |
| Point your index finger up, hand raised | 🐵 Monkey pointing |
| Both hands together | 🐵 Monkey praying |
| Head really tilted + looking side/down | 🟢 Shrek smolder |
| Cover your nose + raise the other hand | 🦊 Nick Wilde, "stinky" |

## Run it

Needs a webcam and Python 3.9 or newer (check that [MediaPipe](https://pypi.org/project/mediapipe/)
supports your Python version).

```bash
pip install -r requirements.txt
python meme_face.py
```

| Key | Does |
|---|---|
| `V` | Tracking overlay: off / lines / mask |
| `C` | Character set: all / baby / monkey / Shrek |
| `Q` | Quit |

## Built with

Python · OpenCV · MediaPipe (Face Landmarker + Hand Landmarker) · NumPy · Pillow

<sub>Meme images and GIFs belong to their respective owners and are used here for fun.</sub>
