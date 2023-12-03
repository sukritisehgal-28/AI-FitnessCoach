import cv2
import mediapipe as mp
import multiprocessing
import numpy as np
import time
from numpy.linalg import norm
from keras.losses import cosine_similarity
mp_drawing = mp.solutions.drawing_utils
mp_drawing2 = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_pose2 = mp.solutions.pose

landmarks_final_cam = []
landmarks_final_video = []

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

cap = cv2.VideoCapture('D:\Studies\Projects\Rock_paper_scissors\hammer_curls.mp4')
cap2 = cv2.VideoCapture(0)
# Curl counter variables
counter = 0 
counter_cam = 0

    
## Setup mediapipe instance
def func1(q):
    with mp_pose2.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap2.isOpened():
            ret2, frame2 = cap2.read()
            
            
            # Recolor image to RGB
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
            # image_cam.flags.writeable =  False

        
            # Make detection
            results2 = pose.process(image2)

        
            # Recolor back to BGR
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                # landmarks_final_2 = []
                landmarks_2 = results2.pose_landmarks.landmark 
                # landmarks_final_cam = []
                # for i in range(len(landmarks_2)):
                #     landmarks_final_cam.append(landmarks_2[i].x)
                #     landmarks_final_cam.append(landmarks_2[i].y)
                #     landmarks_final_cam.append(landmarks_2[i].z)
                # Get coordinates
                shoulder = [landmarks_2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks_2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks_2[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks_2[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks_2[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks_2[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                angle2 = calculate_angle(shoulder, elbow, wrist)
                # Visualize angle
                cv2.putText(image2, str(angle2), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                q.put(angle2)
                # q.put(landmarks2)
            except:
                pass
            
            
            # Render detections
            mp_drawing2.draw_landmarks(image2, results2.pose_landmarks, mp_pose2.POSE_CONNECTIONS,
                                    mp_drawing2.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing2.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
                                        
            
            cv2.imshow('Mediapipe Feed', image2)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
def func2(q):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
            # image_cam.flags.writeable =  False

        
            # Make detection
            results = pose.process(image)

        
            # Recolor back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark 
                # landmarks_final_video = []
                # for i in range(len(landmarks)):
                #     landmarks_final_video.append(landmarks[i].x)
                #     landmarks_final_video.append(landmarks[i].y)
                #     landmarks_final_video.append(landmarks[i].z)
                shoulder_cam = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_cam = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_cam = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angle = calculate_angle(shoulder_cam, elbow_cam, wrist_cam)
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow_cam, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                q.put(angle)
            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )

            cv2.imshow('Mediapipe Feed', image)
            # cv2.imshow('Mediapipe Feed', np.concatenate((cv2.resize(image, (400,400)), cv2.resize(image2,(400,400))),axis = 1))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                # cap2.release()

    # cap.release()
    # cap2.release()

if __name__ == "__main__":
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=func1, args=(q1,))
    p2 = multiprocessing.Process(target=func2, args=(q2,))
    p1.start()
    p2.start()
    print("Starting")
    while True:
        angle1 = q1.get()
        angle2 = q2.get()
        # arr = np.asarray(landmark_2)
        # print(len(landmark_2))
        # print(len(landmark_1))

        # print(landmark_1)
        # print(landmark_2)
        try:
            # print(angle1)
            # print(angle2)
            if angle1>angle2-30  and angle1<angle2 +30:
                print("Keep it going!!")
            else:
                print("Adjust your posture") 
            
        except:
            pass


    #p1.join()
    #p2.join()
    cv2.destroyAllWindows()