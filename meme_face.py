import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import sys
from PIL import Image

# Get base path for bundled files (works for exe and script)
def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

def get_path(filename):
    return os.path.join(BASE_PATH, filename)

# Download model files if not present
def download_model(url, filename):
    filepath = get_path(filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename}")

download_model(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "hand_landmarker.task"
)
download_model(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "face_landmarker.task"
)

# Config
INTRO_VIDEO = get_path("intro.mp4")
INTRO_LOOPS = 1
WINDOW_NAME = "safa_baby"
IMG_SIZE = (400, 400)

# Load images - organized by character set
images = {
    # Baby (default/neutral for all)
    'neutral': cv2.imread(get_path('neutral.png')),
    'baby_hand': cv2.imread(get_path('hand_raised.png')),
    'baby_side': cv2.imread(get_path('look_side.png')),
    'baby_down': cv2.imread(get_path('look_down.png')),
    # Monkey
    'monkey_bite': cv2.imread(get_path('monkey_biting_finger.jpg')),
    'monkey_point': cv2.imread(get_path('monkey_raise_finger.jpg')),
    'monkey_pray': cv2.imread(get_path('monkey_holding_hands.jpg')),
    # Shrek
    'shrek_smolder': cv2.imread(get_path('shrek_smolder.jpeg'))
}

# Resize all images
for key in images:
    if images[key] is not None:
        images[key] = cv2.resize(images[key], IMG_SIZE)

# Load Nick Wilde gif frames
nick_gif_frames = []
try:
    gif = Image.open(get_path('nick-wilde-zootopia.gif'))
    while True:
        frame = gif.copy().convert('RGB')
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, IMG_SIZE)
        nick_gif_frames.append(frame_bgr)
        gif.seek(gif.tell() + 1)
except EOFError:
    pass
except Exception as e:
    print(f"Could not load gif: {e}")

# Character sets mapping pose -> image
CHARACTER_SETS = {
    'baby': {
        'hand_raised': 'baby_hand',
        'look_side': 'baby_side', 
        'look_down': 'baby_down',
        'bite_finger': 'baby_hand',
        'point_up': 'baby_side',
        'praying': 'baby_hand',
        'smolder': 'baby_side',
        'stinky': 'nick_gif',
    },
    'monkey': {
        'hand_raised': 'monkey_pray',
        'look_side': 'monkey_point',
        'look_down': 'monkey_bite',
        'bite_finger': 'monkey_bite',
        'point_up': 'monkey_point',
        'praying': 'monkey_pray',
        'smolder': 'monkey_bite',
        'stinky': 'nick_gif',
    },
    'shrek': {
        'hand_raised': 'shrek_smolder',
        'look_side': 'shrek_smolder',
        'look_down': 'shrek_smolder',
        'bite_finger': 'shrek_smolder',
        'point_up': 'shrek_smolder',
        'praying': 'shrek_smolder',
        'smolder': 'shrek_smolder',
        'stinky': 'nick_gif',
    },
    'all': {
        'hand_raised': 'baby_hand',
        'look_side': 'baby_side',
        'look_down': 'baby_down',
        'bite_finger': 'monkey_bite',
        'point_up': 'monkey_point',
        'praying': 'monkey_pray',
        'smolder': 'shrek_smolder',
        'stinky': 'nick_gif',
    }
}

# Face mesh landmark indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Colors
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
LIME = (0, 255, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BG = (30, 30, 30)

def play_intro():
    """Play intro video looped 3 times"""
    if not os.path.exists(INTRO_VIDEO):
        print(f"Intro video not found: {INTRO_VIDEO}")
        return
    
    print("Playing intro...")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 450)
    
    for loop in range(INTRO_LOOPS):
        cap = cv2.VideoCapture(INTRO_VIDEO)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to fit window
            frame = cv2.resize(frame, (900, 450))
            cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                return False  # User quit
        cap.release()
    
    return True  # Continue to main app

def draw_face_mesh(frame, landmarks, h, w):
    """Draw minimal dots and lines"""
    def get_point(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))
    
    key_points = [
        (61, MAGENTA), (291, MAGENTA), (13, MAGENTA), (14, MAGENTA),
        (33, LIME), (133, LIME), (362, LIME), (263, LIME),
        (1, CYAN), (152, CYAN), (10, CYAN),
    ]
    
    for idx, color in key_points:
        cv2.circle(frame, get_point(idx), 2, color, -1, cv2.LINE_AA)
    
    cv2.line(frame, get_point(61), get_point(291), MAGENTA, 1, cv2.LINE_AA)
    cv2.line(frame, get_point(13), get_point(14), MAGENTA, 1, cv2.LINE_AA)
    cv2.line(frame, get_point(33), get_point(133), LIME, 1, cv2.LINE_AA)
    cv2.line(frame, get_point(362), get_point(263), LIME, 1, cv2.LINE_AA)
    cv2.line(frame, get_point(10), get_point(1), CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, get_point(1), get_point(152), CYAN, 1, cv2.LINE_AA)

def draw_face_mask(frame, landmarks, h, w):
    """Draw full face mesh mask"""
    def get_point(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))
    
    pts = [get_point(i) for i in FACE_OVAL]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i+1) % len(pts)], CYAN, 1, cv2.LINE_AA)
    
    for lip_indices in [LIPS_OUTER, LIPS_INNER]:
        pts = [get_point(i) for i in lip_indices]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i+1) % len(pts)], MAGENTA, 1, cv2.LINE_AA)
    
    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        pts = [get_point(i) for i in eye_indices]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i+1) % len(pts)], LIME, 1, cv2.LINE_AA)

def draw_hand_landmarks(frame, hand_landmarks, h, w):
    """Draw minimal hand"""
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    fingertips = [4, 8, 12, 16, 20]
    
    cv2.circle(frame, points[0], 3, CYAN, -1, cv2.LINE_AA)
    for tip in fingertips:
        cv2.circle(frame, points[tip], 2, MAGENTA, -1, cv2.LINE_AA)
        cv2.line(frame, points[0], points[tip], LIME, 1, cv2.LINE_AA)

def is_hand_near_face(hand_result, face_result):
    if not hand_result.hand_landmarks or not face_result.face_landmarks:
        return False, False, False, False, 0, False
    
    nose = face_result.face_landmarks[0][1]
    chin = face_result.face_landmarks[0][152]
    mouth_y = (nose.y + chin.y) / 2
    
    hand_near = False
    finger_extended_near_face = False
    pointing_up = False
    hands_together = False
    nose_cover_and_wave = False
    num_hands = len(hand_result.hand_landmarks)
    
    hands = hand_result.hand_landmarks
    
    # Check if two hands are together (praying)
    if num_hands >= 2:
        hand1_center = hands[0][9]
        hand2_center = hands[1][9]
        dist_between_hands = ((hand1_center.x - hand2_center.x)**2 + (hand1_center.y - hand2_center.y)**2)**0.5
        if dist_between_hands < 0.2:
            hands_together = True
            return hand_near, finger_extended_near_face, pointing_up, hands_together, num_hands, nose_cover_and_wave
        
        # Check for nose cover + wave (one hand near nose, other hand away/waving)
        hand1_to_nose = ((hands[0][9].x - nose.x)**2 + (hands[0][9].y - nose.y)**2)**0.5
        hand2_to_nose = ((hands[1][9].x - nose.x)**2 + (hands[1][9].y - nose.y)**2)**0.5
        
        # One hand close to nose (pinching), other hand anywhere else
        if hand1_to_nose < 0.2 and hand2_to_nose > 0.15:
            nose_cover_and_wave = True
        elif hand2_to_nose < 0.2 and hand1_to_nose > 0.15:
            nose_cover_and_wave = True
        
        if nose_cover_and_wave:
            return hand_near, finger_extended_near_face, pointing_up, hands_together, num_hands, nose_cover_and_wave
    
    for hand in hands:
        wrist = hand[0]
        index_tip = hand[8]
        index_base = hand[5]
        middle_tip = hand[12]
        middle_base = hand[9]
        
        # Check if hand near face
        for idx in [0, 5, 9, 13, 17]:
            point = hand[idx]
            dist = ((point.x - nose.x)**2 + (point.y - nose.y)**2)**0.5
            if dist < 0.25:
                hand_near = True
        
        # Check if finger is extended (tip above base = open finger)
        index_extended = index_tip.y < index_base.y - 0.02
        middle_extended = middle_tip.y < middle_base.y - 0.02
        any_finger_extended = index_extended or middle_extended
        
        # Check if extended finger is near face
        if any_finger_extended and hand_near:
            finger_extended_near_face = True
        
        # Check if pointing up (index tip above index base, and hand raised high)
        if index_tip.y < index_base.y - 0.1 and index_tip.y < 0.4:
            middle_closed = hand[12].y > hand[9].y
            ring_closed = hand[16].y > hand[13].y
            pinky_closed = hand[20].y > hand[17].y
            if middle_closed and ring_closed and pinky_closed:
                pointing_up = True
    
    return hand_near, finger_extended_near_face, pointing_up, hands_together, num_hands, nose_cover_and_wave

def get_head_pose(face_result):
    if not face_result.face_landmarks:
        return None
    
    lm = face_result.face_landmarks[0]
    nose, left_eye, right_eye = lm[1], lm[33], lm[263]
    forehead, chin = lm[10], lm[152]
    left_ear, right_ear = lm[234], lm[454]
    
    eye_width = right_eye.x - left_eye.x
    head_turn = (nose.x - (left_eye.x + right_eye.x) / 2) / eye_width if eye_width > 0 else 0
    
    face_height = chin.y - forehead.y
    head_tilt = (nose.y - forehead.y) / face_height if face_height > 0 else 0.5
    
    # Head roll (tilt left/right) - compare ear heights
    head_roll = right_ear.y - left_ear.y  # Positive = tilted right, Negative = tilted left
    
    # Looking up detection - forehead more visible, chin less
    looking_up = head_tilt < 0.45
    
    return {'turn': head_turn, 'tilt': head_tilt, 'roll': head_roll, 'looking_up': looking_up}

def get_face_expression(face_result):
    expr = {'mouth_open': False, 'tongue_out': False}
    
    if not face_result.face_blendshapes:
        return expr
    
    for bs in face_result.face_blendshapes[0]:
        if bs.category_name == 'jawOpen' and bs.score > 0.15:
            expr['mouth_open'] = True
        if bs.category_name == 'tongueOut' and bs.score > 0.1:
            expr['tongue_out'] = True
    
    if expr['mouth_open']:
        for bs in face_result.face_blendshapes[0]:
            if bs.category_name == 'jawOpen' and bs.score > 0.4:
                expr['tongue_out'] = True
    
    return expr

def draw_status_bar(frame, pose, character, draw_mode, w):
    """Draw minimal status bar at top"""
    cv2.rectangle(frame, (0, 0), (w, 35), DARK_BG, -1)
    
    state_colors = {
        'neutral': (150, 150, 150), 'hand_raised': MAGENTA, 'look_side': CYAN, 
        'look_down': ORANGE, 'bite_finger': LIME, 'point_up': (0, 200, 255), 
        'praying': (255, 200, 0), 'smolder': (0, 255, 100), 'stinky': (255, 100, 50)
    }
    state_names = {
        'neutral': 'NEUTRAL', 'hand_raised': 'HAND UP', 'look_side': 'LOOK SIDE', 
        'look_down': 'LOOK DOWN', 'bite_finger': 'BITE FINGER', 'point_up': 'POINT UP', 
        'praying': 'PRAYING', 'smolder': 'SMOLDER', 'stinky': 'STINKY'
    }
    
    # Pose name
    cv2.putText(frame, state_names.get(pose, ''), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_colors.get(pose, WHITE), 1, cv2.LINE_AA)
    
    # Character mode
    cv2.putText(frame, f"[C] {character}", (w//2 - 40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    
    # View mode
    mode_names = ['OFF', 'LINES', 'MASK']
    cv2.putText(frame, f"[V] {mode_names[draw_mode]}", (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

def main():
    # Play intro
    if not play_intro():
        return
    
    print("\nLoading main app...")
    
    # Create detectors
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=get_path('hand_landmarker.task')),
            num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    )
    
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=get_path('face_landmarker.task')),
            num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
            output_face_blendshapes=True
        )
    )
    
    cap = cv2.VideoCapture(0)
    current_pose = 'neutral'
    draw_mode = 1
    character_mode = 0  # 0=all, 1=baby, 2=monkey, 3=shrek
    character_names = ['ALL', 'BABY', 'MONKEY', 'SHREK']
    character_keys = ['all', 'baby', 'monkey', 'shrek']
    
    # Nick gif state
    playing_nick_gif = False
    nick_gif_frame_idx = 0
    nick_gif_loops = 0
    nick_gif_max_loops = 3
    
    print("\n=== SAFA BABY ===")
    print("[V] Toggle view  |  [C] Toggle character  |  [Q] Quit\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Resize camera to match baby image height
        cam_h = IMG_SIZE[1]
        cam_w = int(w * cam_h / h)
        frame = cv2.resize(frame, (cam_w, cam_h))
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_result = hand_detector.detect(mp_image)
        face_result = face_detector.detect(mp_image)
        
        # Draw tracking
        if draw_mode > 0:
            if face_result.face_landmarks:
                if draw_mode == 1:
                    draw_face_mesh(frame, face_result.face_landmarks[0], h, w)
                else:
                    draw_face_mask(frame, face_result.face_landmarks[0], h, w)
            if hand_result.hand_landmarks:
                for hand in hand_result.hand_landmarks:
                    draw_hand_landmarks(frame, hand, h, w)
        
        # Detect state
        hand_near, finger_extended_near_face, pointing_up, hands_together, num_hands, nose_cover_wave = is_hand_near_face(hand_result, face_result)
        head_pose = get_head_pose(face_result)
        expr = get_face_expression(face_result)
        
        # Check for stinky gesture (nose cover + wave) - triggers gif
        if nose_cover_wave and not playing_nick_gif and len(nick_gif_frames) > 0:
            playing_nick_gif = True
            nick_gif_frame_idx = 0
            nick_gif_loops = 0
            current_pose = 'stinky'
        
        # If playing gif, continue until done
        if playing_nick_gif:
            current_pose = 'stinky'
        # STINKY has priority when 2 hands detected
        elif nose_cover_wave:
            current_pose = 'stinky'
        # HAND GESTURES (highest priority - 2 hands first!)
        elif hands_together and num_hands >= 2:
            current_pose = 'praying'
        elif pointing_up and not finger_extended_near_face:
            current_pose = 'point_up'
        elif finger_extended_near_face:
            current_pose = 'bite_finger'
        elif hand_near and not finger_extended_near_face:
            current_pose = 'hand_raised'
        # FACE ONLY GESTURES (no hands involved)
        elif head_pose:
            looking_side = abs(head_pose['turn']) > 0.18
            looking_down = head_pose['tilt'] > 0.58
            head_tilted = abs(head_pose['roll']) > 0.05
            head_really_tilted = abs(head_pose['roll']) > 0.08  # Really tilted to one side
            looking_really_side = abs(head_pose['turn']) > 0.25  # Really turned
            
            if head_really_tilted and (looking_really_side or looking_down):
                # Head really tilted + looking down or really to side = smolder
                current_pose = 'smolder'
            elif looking_down and expr['tongue_out'] and not looking_side:
                current_pose = 'look_down'
            elif looking_side and expr['mouth_open'] and not looking_down:
                current_pose = 'look_side'
            else:
                current_pose = 'neutral'
        else:
            current_pose = 'neutral'
        
        # Get the right image based on pose and character mode
        if playing_nick_gif and len(nick_gif_frames) > 0:
            # Play gif frame
            display_img = nick_gif_frames[nick_gif_frame_idx]
            nick_gif_frame_idx += 1
            if nick_gif_frame_idx >= len(nick_gif_frames):
                nick_gif_frame_idx = 0
                nick_gif_loops += 1
                if nick_gif_loops >= nick_gif_max_loops:
                    playing_nick_gif = False
                    current_pose = 'neutral'
        elif current_pose == 'neutral':
            display_img = images['neutral']
        else:
            char_key = character_keys[character_mode]
            img_key = CHARACTER_SETS[char_key].get(current_pose, 'neutral')
            display_img = images.get(img_key, images['neutral'])
        
        # Draw status bar on camera feed
        draw_status_bar(frame, current_pose, character_names[character_mode], draw_mode, w)
        
        # Get baby image
        baby_img = display_img
        if baby_img is None:
            baby_img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        
        # Combine side by side: Baby | Camera
        combined = np.hstack([baby_img, frame])
        
        cv2.imshow(WINDOW_NAME, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            draw_mode = (draw_mode + 1) % 3
        elif key == ord('c'):
            character_mode = (character_mode + 1) % 4
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
