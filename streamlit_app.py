import cv2
import numpy as np
import pandas as pd
import streamlit as st

# define the weights to be used along with its config file
config =  'yolov3.cfg'
wt_file = 'yolov3.weights'

# Add a title and sidebar
st.title("Object Detection")
st.sidebar.markdown("# Model")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

@st.cache(show_spinner=True)
def read_img(img):
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image


def yolo_v3(image, confidence_threshold=0.5, overlap_threshold=0.3):
    # Load model architecture
    net = cv2.dnn.readNetFromDarknet(config, wt_file)
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Set input and get output
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]

    # For each detected object, compute the box, find the score, ignore if below
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)

    # Write the name of detected objects above image
    f = open("classes.txt", "r")
    f = f.readlines()
    f = [line.rstrip('\n') for line in list(f)]
    try:
        st.subheader("Detected objects: " + ', '.join(list(set([f[obj] for obj in class_IDs]))))
    except IndexError:
        st.write("Nothing detected")

    # Apply non-max suppression to identify best bounding box
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            xmin.append(x)
            ymin.append(y)
            xmax.append(x + w)
            ymax.append(y + h)
    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    # Add a layer on top on a detected object
    LABEL_COLORS = [0, 255, 0]
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += LABEL_COLORS
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    # Display the final image
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

