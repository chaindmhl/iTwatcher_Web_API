import cv2
import numpy as np

def perform_yolov4_inference(image_path, weights_path = "/home/icebox/itwatcher_api/darknet/LPR/backup_2/yolov4-custom_final.weights", config_path = "/home/icebox/itwatcher_api/darknet/LPR/yolov4-custom.cfg", class_names_path = "/home/icebox/itwatcher_api/darknet/LPR/obj.names", confidence_threshold=0.5, nms_threshold=0.4):
    # Load YOLOv4 model
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load class names
    with open(class_names_path, 'r') as f:
        classes = f.read().strip().split('\n')
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform inference
    outputs = net.forward(output_layers)
    
    # Post-process results
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
        
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
        
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    # Draw bounding boxes
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

# Paths to files
image_path = 'path_to_your_image.jpg'
#weights_path = 'yolov4.weights'
#config_path = 'yolov4.cfg'
#class_names_path = 'coco.names'

# Perform inference and get the annotated image
annotated_image = perform_yolov4_inference("/home/icebox/itwatcher_api/plates/3.jpg")

# Display the annotated image
cv2.imshow('YOLOv4 Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
