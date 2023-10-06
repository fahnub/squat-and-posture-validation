from sys import platform
import numpy as np
import argparse
import sys
import cv2


def calculate_angle(a,b,c):
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


try:    
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op

    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="test.png", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    fcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (1920, 1080)
    video_number = 4
    path = f"media/squat-good/angle-1/{video_number}"
    video_output = cv2.VideoWriter(f"{path}.avi", fcc, 60, size)
    cap = cv2.VideoCapture(f"{path}.mp4")

    while(True):
        
        ret, frame = cap.read()

        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is not None:

            pose_keypoints = datum.poseKeypoints.tolist()[0]

            r_knee = datum.poseKeypoints[0][10]
            r_knee = r_knee[0:2]

            l_knee = datum.poseKeypoints[0][13]
            l_knee = r_knee[0:2]

            right_hip = pose_keypoints[9]

            left_hip = pose_keypoints[12]

            right_knee = pose_keypoints[10]

            left_knee = pose_keypoints[13]

            right_foot = pose_keypoints[11]

            left_foot = pose_keypoints[14]

            r_knee_angle = calculate_angle(right_hip, right_knee, right_foot)
            l_knee_angle = calculate_angle(left_hip, left_knee, left_foot)
            angle = (r_knee_angle + l_knee_angle) / 2

            frame = datum.cvOutputData
        
            # cv2.putText(frame, str(angle), (int(r_knee[0]), int(r_knee[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Squat Angle: {str(angle)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)

            if angle <= 90:
                cv2.putText(frame, f"SQUAT GOOD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
            elif angle > 90:
                cv2.putText(frame, f"SQUAT BAD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
        
        else:
            continue
        
        video_output.write(frame)
        cv2.imshow("OpenPose Output", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    video_output.release()

except Exception as e:
    print(e)
    sys.exit(-1)
