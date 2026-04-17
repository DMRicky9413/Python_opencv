import cv2
import numpy as np

confThreshold = 0.8
cap = cv2.VideoCapture(0)

classesFile = 'yolov3 model/coco80.names'
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

fruit_classes = ['apple', 'banana', 'orange']
fruit_prices = {'apple': 5, 'banana': 3, 'orange': 4,}

net = cv2.dnn.readNetFromDarknet('yolov3 model/yolov3-608.cfg','yolov3 model/yolov3-608.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success , img = cap.read()
    height, width, ch = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    LayerOutputs = net.forward(output_layers_names)

    bboxes, confidences, class_ids = [], [], []

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confThreshold and classes[class_id] in fruit_classes:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bboxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0,255,size=(len(bboxes),3))

    fruit_counts = {fruit: 0 for fruit in fruit_classes}
    total_count = 0
    total_price = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = bboxes[i]
            label = classes[class_ids[i]]
            conf = confidences[i]
            color = colors[i]

            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,f"{label} {conf:.2f}",(x,y-10),font,0.6,(255,255,255),2)

            fruit_counts[label] += 1
            total_count += 1
            total_price += fruit_prices[label]

    cv2.putText(img, f"Total: {total_count}  Price: ${total_price}",
                (width-500, 50), font, 1.5, (0,255,255), 2)

    y_offset = 100
    for fruit, count in fruit_counts.items():
        if count > 0:
            cv2.putText(img, f"{fruit}: {count}",
                        (width-500, y_offset), font, 1.5, (0,255,0), 2)
            y_offset += 50

    cv2.imshow('Fruit', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
