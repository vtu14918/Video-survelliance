import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys

confid = 0.5
thresh = 0.5

vid_path = "./videos/video4.mp4"


# Calibration needed for each video

def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0

def object_detection_video():
    labelsPath = "./coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"

###### use this for faster processing (caution: slighly lower accuracy) ###########

# weightsPath = "./yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
# configPath = "./yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    st.title("Object Detection for Videos")
    st.subheader("""
    This object detection project takes in a video and outputs the video with bounding boxes created around the objects in the video 
    """
    )

    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        #video_file = 'street.mp4'
        vs = cv2.VideoCapture(vid)
        _, frame = vs.read()
        writer = None
        (W, H) = (None, None)
        q = 0
        count=0
        while True:
           _, frame = vs.read()
           if _ != False:
                H, W = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                #Blob stands for Binary Large Object and refers to the connected pixel in the binary image. 
                #The term "Large" focuses on the object of a specific size, and that other "small" binary objects are usually noise.
                net.setInput(blob)
                start = time.perf_counter()
                layerOutputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count +=1
                print(f"Time took: {count}", time_took)

                boxes = []
                confidences = []
                classIDs = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)#finding largest predicted probabaility
                        confidence = scores[classID]
                        if LABELS[classID] == "person":
                            if confidence > confid:
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                #confid= st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
                #thresh= st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.5, 0.01)
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

                if len(idxs) > 0:
                     status = list()
                     idf = idxs.flatten()
                     close_pair = list()
                     s_close_pair = list()
                     center = list()
                     dist = list()
                     for i in idf:
                         (x, y) = (boxes[i][0], boxes[i][1])
                         (w, h) = (boxes[i][2], boxes[i][3])
                         center.append([int(x + w / 2), int(y + h / 2)])
                         status.append(0)
                     for i in range(len(center)):
                        for j in range(len(center)):
                            g = isclose(center[i], center[j])

                            if g == 1:
                                 close_pair.append([center[i], center[j]])
                                 status[i] = 1
                                 status[j] = 1
                            elif g == 2:
                                s_close_pair.append([center[i], center[j]])
                                if status[i] != 1:
                                    status[i] = 2
                                if status[j] != 1:
                                   status[j] = 2

                     total_p = len(center)
                     low_risk_p = status.count(2)
                     high_risk_p = status.count(1)
                     safe_p = status.count(0)
                     kk = 0

                     for i in idf:
                        sub_img = frame[10:170, 10:W - 10]
                        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
                        res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 1.0)
                        frame[10:170, 10:W - 10] = res
                        cv2.putText(frame, "Video Surveillance - Distant Socializing", (210, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.rectangle(frame, (20, 60), (510, 160), (170, 170, 170), 2)
                        cv2.putText(frame, "Connecting lines shows closeness among people. ", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        cv2.putText(frame, "-- YELLOW: CLOSE", (50, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(frame, "--    RED: VERY CLOSE", (50, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        

                        cv2.rectangle(frame, (535, 60), (W - 20, 160), (170, 170, 170), 2)
                        cv2.putText(frame, "Bounding box shows the level of risk to the person.", (545, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        cv2.putText(frame, "-- DARK RED: HIGH RISK", (565, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
                        cv2.putText(frame, "--   ORANGE: LOW RISK", (565, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

                        cv2.putText(frame, "--    GREEN: SAFE", (565, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        tot_str = "TOTAL COUNT: " + str(total_p)
                        high_str = "HIGH RISK COUNT: " + str(high_risk_p)
                        low_str = "LOW RISK COUNT: " + str(low_risk_p)
                        safe_str = "SAFE COUNT: " + str(safe_p)

                        sub_img = frame[H - 120:H, 0:210]
                        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

                        res = cv2.addWeighted(sub_img, 0.8, black_rect, 0.2, 1.0)

                        frame[H - 120:H, 0:210] = res

                        cv2.putText(frame, tot_str, (10, H - 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(frame, safe_str, (10, H - 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        cv2.putText(frame, low_str, (10, H - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 1)
                        cv2.putText(frame, high_str, (10, H - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)

                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        if status[kk] == 1:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

                        elif status[kk] == 0:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

                        kk += 1
                     for h in close_pair:
                         cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
                     for b in s_close_pair:
                         cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

                     cv2.imshow('Video Surveillance', frame)
                     if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MPV4")
                    writer = cv2.VideoWriter("output.mp4", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
                
                writer.write(frame)
        print("Processing finished: open output.mp4")
        print("HURRAY!! WE SUCCESSFULLY SURVELLIANCE THE PHYSICAL DISTANCE")
        writer.release()
        vs.release()
def upload_image_ui():
    uploaded_image = st.file_uploader("Please choose an image file", type=["png", "jpg", "jpeg","jfif"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
        except Exception:
            st.error("Error: Invalid image")
        else:
            img_array = np.array(image)
            return img_array
def detect_object(frame):
    cfg_path = os.path.abspath('./yolov3.cfg')
    weights_path = os.path.abspath('./yolov3.weights')
    names_path = os.path.abspath('./coco.names')

    # Load Yolo
    # net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
    net = cv2.dnn_DetectionModel(cfg_path, weights_path)
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    # frame = cv2.imread('sample1.jpg')
    # print(type(frame))
    # Resize the image
    frame = cv2.resize(frame, dsize=(704, 704), interpolation=cv2.INTER_AREA)

    # with open('coco.names', 'rt') as f:
    with open(names_path, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    # print("Cl:", classes)
    # print("Co:", confidences)
    # print("Clf:", classes.flatten())
    # print("Cof:", confidences.flatten())
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        # print(classId, confidence, box)
        label = '%.2f' % confidence
        label = '%s: %s' % (names[classId], label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # fontScale: 0.5, thickness: 1
        # print(labelSize, baseLine)
        left, top, width, height = box
        # print("T:", top)
        top = max(top, labelSize[1])
        # print("MT:", top)
        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
        # Draw rectangle for labels
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame
    # cv2.imshow('out', frame)
    # cv2.waitKey()
def main():
    
    image = Image.open('1st image.jpg')

    st.image(image, caption='Yolo object detection architecture')
    new_title = '<p style="font-size: 49px;">DISTANCE SOCIALIZING SURVEILLANCE USING YOLOV3 ALGORITHM </p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate YOLO Object detection in both videos(pre-recorded)
    and images.
    
    
    This YOLO object Detection project can detect 80 objects(i.e classes)
    in either a video or image. 
    Object Recognition using Yolo is the application which can recognize the objects in a video stream. The model can
    classify the object in the image up to 85 percent accuracy.
    This takes in the input as an image or video element and can predict the objects in this. “You Only Look Once,” or
    YOLO, family of Convolutional Neural Networks that achieve near state-of-the-art results with a single end-to-end
    model that can perform object detection in real-time.
    Unlike another image classifier this can recognize an array of multiple objects in a single instance of frame from an
    image or video. The approach involves a single deep convolutional neural network (originally a version of
    GoogLeNet that splits the input into a grid of cells and each cell directly predicts a bounding box and object
    classification.""")
    read_me_1=st.header("**Aim of the Project:**")
    read_me_2=st.markdown("""The main aim of the project is that to use the yolo model for better surveillance of the video""")
    read_me_3=st.header("**Scope of the Project:**")
    read_me_4=st.markdown("""It is a challenging problem that involves building upon methods for object recognition (e.g. where are they), object
    localization (e.g. what are their extent), and object classification (e.g. what are they)
    The main objective is to build a object detection model to predict with higher rate of accuracy
    Nowadays monitoring is a subject of interest from video surveillance systems especially for those one that set their
    goals to monitor people.
    As an example of using these methods one can suggest different security video systems which aim to detect violators
    in restricted areas, video monitoring in airports, public spaces, roads etc.
    Therefore the problem of efficient monitoring is important and can be applied to an abundance of different tasks.
    However, in all these applications we have to be faced with problems connected with bad quality images, sufficient
    enclosures of objects by the foreground, strict conditions imposed on computational and time resources given for
    particular algorithms.
    These restrictions influence the algorithm's quality as negative factors reduce its efficiency calculated in assumption
    to ideal data.""")
    read_me_5=st.header("**MODULES ARE:**")
    """ *MODULE 1-- Image Upload – It allows us to upload images into the model
    *MODULE 2--Camera Module– It takes the input from the live camera feed
    *MODULE 3--Yolo Model – Process the object based on bounding boxes
    *MODULE 4--Processing model – Where distance between two persons are detected"""
    read_me_6=st.header("**Conclusion:**")
    """**To generate correct social gatherings of humans from RGB channel pictures containing localized objects.
    Additionally the matter of inaccurate predictions has been resolved and a correct bounding box has been
    drawn round the divided region.**"""

    st.sidebar.title("Select Activity ")
    st.sidebar.image("yolo.jpg", use_column_width=True)
    choice  = st.sidebar.selectbox("MODE",("About","Object Detection(Image)","Social Distance Object Detection(Video)"))
    #["Show Instruction","Landmark identification","Show the #source code", "About"]
    
    if choice == "Object Detection(Image)":
        #st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('Object Detection')
        st.title("Object detection with YOLOv4")
        img_array = upload_image_ui()

        if isinstance(img_array, np.ndarray):
            image = detect_object(img_array)
            st.image(image)
        
    elif choice == "Social Distance Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()
        read_me_1.empty()
        read_me_2.empty()
        read_me_3.empty()
        read_me_4.empty()
          #object_detection_video.has_beenCalled = False
        object_detection_video()
        #st.open(jh,'rb')
        #st.image(jh)
        #if object_detection_video.has_beenCalled:
        try:

            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video") 
        except OSError:
            ''

    elif choice == "About":
        print()
        

if __name__ == '__main__':
		main()	

