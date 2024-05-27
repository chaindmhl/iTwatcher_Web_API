import cv2
import numpy as np

class YOLOv4Inference:
    def __init__(self, weights_path = "/home/icebox/itwatcher_api/tracking/deepsort_tric/data/ocr_comb.weights", config_path = "/home/icebox/itwatcher_api/darknet/LPR/yolov4-custom.cfg", class_names_path = "/home/icebox/itwatcher_api/darknet/LPR/obj.names", confidence_threshold=0.5, nms_threshold=0.4):
    # Load YOLOv4 model, config_path, class_names_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.weights_path = weights_path
        self.config_path = config_path
        self.class_names_path = class_names_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        with open(self.class_names_path, 'r') as f:
            self.classes = f.read().strip().split('\n')
        
    def infer_image(self, img):
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outputs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
            
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
            
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
        
        detected_classes = []
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[i]
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_classes.append((label, x))

        return img,detected_classes
    
    def infer_image_only(self, img):
        _, detected_classes_with_x = self.infer_image(img)

        # Sort the detected classes based on their x-coordinates
        sorted_detected_classes = [label for label, _ in sorted(detected_classes_with_x, key=lambda x: x[1])]

        return {"detected_classes": sorted_detected_classes}


    def infer_image_only_thresh(self, img):
        prediction = self.infer_image(img)

        if prediction is not None:
            _, detected_classes_with_x = prediction

            # Sort the detected classes based on their x-coordinates
            sorted_detected_classes = [label for label, _ in sorted(detected_classes_with_x, key=lambda x: x[1])]

            # Check if the number of detected classes is in the allowed lengths
            num_detected_classes = len(sorted_detected_classes)
            if num_detected_classes in [6, 7, 10]:
                return {"detected_classes": sorted_detected_classes, "within_range": True}
            else:
                return {"detected_classes": [], "within_range": False, "reason": "Invalid number of detected classes"}

        return {"detected_classes": [], "within_range": False}



       
    def infer_and_save(self, plate_image):
        # Perform inference and get the detected classes
        _, detected_classes_with_x = self.infer_image(plate_image)
        
        # Sort the detected classes based on their x-coordinates
        sorted_detected_classes = [label for label, _ in sorted(detected_classes_with_x, key=lambda x: x[1])]
        
        # Generate the output filename based on sorted detected classes (from left to right)
        output_filename = "".join(sorted_detected_classes) + '.jpg'
        
        output_path = output_filename
        
        # Save the plate image (no need for the annotated image if not necessary)
        cv2.imwrite(output_path, plate_image)

        return {"detected_classes": sorted_detected_classes}


