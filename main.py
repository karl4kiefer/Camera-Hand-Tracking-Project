import cv2
import time
import mediapipe as mp
import numpy as np



# camera 

def main():
    #cam
    cam = cv2.VideoCapture(0)
    cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (cam_w, cam_h))

    #hand
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    start = time.time()    
    print("Starting main loop")
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            
            out.write(frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            print(result.hand_landmarks)

            if result.hand_landmarks:
                draw_hands(result, frame)
            
            cv2.imshow('Camera', frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        
    cam.release()
    out.release()
    cv2.destroyAllWindows()


def draw_hands(result, frame):
    h, w, _ = frame.shape
    thumb_tip, pointer_tip = result.hand_landmarks[0][4], result.hand_landmarks[0][8]
    cv2.circle(frame, (int(w * result.hand_landmarks[0][8].x), int(h * result.hand_landmarks[0][8].y)), 50, (125, 0, 130), -1)
    cv2.circle(frame, (int(w * result.hand_landmarks[0][4].x), int(h * result.hand_landmarks[0][4].y)), 50, (125, 0, 130), -1)

if __name__ == "__main__":
    main()

