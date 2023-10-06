from sys import platform
import numpy as np
import argparse
import sys
import cv2

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
    video_number = 5
    path = f"media/posture-good/{video_number}"
    video_output = cv2.VideoWriter(f"{path}.avi", fcc, 60, size)
    cap = cv2.VideoCapture(f"{path}.mp4")

    while(True):
        ret, frame = cap.read()

        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is not None:

            pose_keypoints = datum.poseKeypoints.tolist()[0]

            hip = datum.poseKeypoints[0][8]
            hip = hip[0:2]

            neck = datum.poseKeypoints[0][1]
            neck = neck[0:2]

            frame = datum.cvOutputData

            cv2.putText(frame, f"Difference: {str(neck[0]-hip[0])}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)

            if neck[0] >= hip[0]:
                cv2.putText(frame, f"POSTURE GOOD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"POSTURE BAD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
        
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