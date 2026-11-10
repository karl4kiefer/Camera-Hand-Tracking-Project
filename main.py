import cv2
import time
import math
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
            
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            
            out.write(frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

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
    size = 10
    thickness = 3
    color = (125, 0, 130)

    index_pairs = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
    for landmark in result.hand_landmarks:
        points = []
        hand = [landmark[0], landmark[1], landmark[2], landmark[3], landmark[4], landmark[5], landmark[6], landmark[7], landmark[8], landmark[9], landmark[10], landmark[11], landmark[12], landmark[13], landmark[14], landmark[15], landmark[16], landmark[17], landmark[18], landmark[19], landmark[20]]

        for index in hand:
            points.append((int(index.x * w), int(index.y * h)))
            cv2.circle(frame, (int(w * index.x), int(h * index.y)), size, color, -1)
            
        for a, b in index_pairs:
            cv2.line(frame, points[a], points[b], color, thickness)



if __name__ == "__main__":
    main()

