import cv2
import numpy as np

# Output Video
video = cv2.VideoWriter('output.mp4', -1, 1, (720, 720))

# Loading Yolo Pretrained CNN
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:  # Read in the training class labels
    classes = [line.strip() for line in f.readlines()]
output_layers = net.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture('videoTest.mp4')  # Test Data (Shortened to two minutes due to memory limitations)
i = 0
while (cap.isOpened()):  # For each frame in the image...
    # Loading image
    ret, img = cap.read()
    if ret == False:
        break

    img = cv2.resize(img, None, fx=0.4, fy=0.4)  # resize to feed into the CNN
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Feed in the test frame to
    # the CNN
    net.setInput(blob)
    outs = net.forward(output_layers)  # Propagate through the network

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Get the 5 best detection probabilities
            class_id = np.argmax(scores)  # Extract the class with the max score
            confidence = scores[class_id]
            # For the purpose of this task, only cars and trucks will be annotated in the image
            if confidence > 0.5 and (classes[class_id] == "car" or classes[class_id] == "truck"):
                # Object detected
                center_x = int(detection[0] * width)  # Find rectangle coordinates from the yolo detector.
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  # Determine rectangle area.
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)  # save the class id for text inside boxes

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Thresholds overlapping boxes in order
    # to determine whether they are detecting the same object and that one should be removed
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, label, (x, y + 30), font, 3, (57, 255, 20), 3)  # Place text inside box
    i += 1  # Iterate through each box
    video.write(img)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cap.release()
cv2.destroyAllWindows()
print("Exited")
