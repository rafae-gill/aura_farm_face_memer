# Aura Farm Meme Face FR FR 👶🐵

A fun face/gesture tracking app that mirrors your expressions with baby, monkey, and Shrek images!

## Features

- Real-time face and hand tracking using MediaPipe
- Multiple character modes: Baby, Monkey, Shrek, or All
- Various gesture triggers:
  - **Neutral** - Normal face → Baby
  - **Hand raised** - Fist near face → Baby with hand
  - **Look side** - Turn head + open mouth → Baby looking side
  - **Look down** - Tilt down + tongue out → Baby looking down
  - **Bite finger** - Finger extended near face → Monkey biting finger
  - **Point up** - Point finger up → Monkey pointing
  - **Praying** - Both hands together → Monkey holding hands
  - **Smolder** - Head really tilted + looking side/down → Shrek smolder
  - **Stinky** - Cover nose + raise other hand → Nick Wilde gif

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python meme_face.py
```

### Controls
- **V** - Toggle tracking view (Off / Lines / Mask)
- **C** - Toggle character mode (All / Baby / Monkey / Shrek)
- **Q** - Quit

PRESS Q to exit on the app

## Requirements

- Python 3.8+
- Webcam

## Credits

Made with using MediaPipe for face and hand tracking.
