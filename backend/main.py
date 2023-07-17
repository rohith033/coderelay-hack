import cv2 as cv
import numpy as np
import mediapipe as mp
import time
timer = time.time()
def Angle(a1,b1,c1, w, h):
    a = np.array([a1.x*w, a1.y*h, a1.z])
    b = np.array([b1.x*w, b1.y*h, b1.z])
    c = np.array([c1.x*w, c1.y*h, c1.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.abs(angle*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
        
    return angle

def squat_check(filename):
    cap = cv.VideoCapture(filename)
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    output={'Errors':[]}
    fourcc = cv.VideoWriter_fourcc(*'MP4V') 
    out = cv.VideoWriter('stats.mp4', fourcc, 20.0, (width,  height))
    # out = cv.VideoWriter('s1.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (640,854))
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    BG_COLOR = (200,200,200)
    InAnkleAngle=0
    InHipAngle=0
    InKneeAngle=0
    reps=0
    currA=0
    currH=0
    currK=0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        counted=0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image)
            image_hight, image_width, _ = image.shape
            kneeAngle=Angle( results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL], image_width, image_hight)
            ankleAngle=Angle( results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL],  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX], image_width, image_hight)
            hipAngle=Angle( results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE], image_width, image_hight)
            if(kneeAngle>=170 and kneeAngle<=180 and counted!=1):
                currA=0
                currH=0
                currK=0
                reps+=1
                counted=1
            if(counted==1 and kneeAngle>=80 and kneeAngle<=100):
                counted=0
            if(abs(kneeAngle-hipAngle)>25):
                InHipAngle=1
            if(ankleAngle>=100 or ankleAngle<80):
                InAnkleAngle=1
            if(kneeAngle<70):
                InKneeAngle=1
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f'left leg knee{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]}',(50,70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
            cv.putText(image, f'left leg hip{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]}',(50,90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
            cv.putText(image, f'left leg ankle{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]}',(50,110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
            cv.putText(image, f'ankle angle{ankleAngle}',(50,150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
            cv.putText(image, f'knee angle{kneeAngle}',(50,130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
            cv.putText(image, f'hip angle{hipAngle}',(50,170), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            out.write(image)
            cv.imshow('MediaPipePose', image)
            if cv.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv.destroyAllWindows()
        if(InAnkleAngle and currA!=1):
            currA=1
            output['Errors'].append("The ankles must be positioned properly within an angle of 80 to 100 while doing squats, your ankle movements is wrong please check")
        if(InHipAngle and currH!=1):
            currH=1
            output['Errors'].append("The hip must be raised and lowered parallel to the knees while doing squats, your hip movement is wrong please check")
        if(InKneeAngle and currK!=1):
            currK=1
            output['Errors'].append("Folding your knees by less than 70 degrees will have have bad impact on knees while doing squats, please check your knee movement")
    print(reps)
    return output  
    
print(squat_check('vid5.mp4'))