import cv2
import numpy as np
import os

class YOLOv4Inference:
    def __init__(self, weights_path="/home/icebox/itwatcher_api/darknet/color/color.weights",
                 config_path="/home/icebox/itwatcher_api/darknet/color/yolov4-custom.cfg",
                 class_names_path="/home/icebox/itwatcher_api/darknet/color/color.names",
                 confidence_threshold=0.5, nms_threshold=0.4):
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

        detected_classes = []
        if indexes is not None and len(indexes) > 0:
            for i in indexes:
                label = str(self.classes[class_ids[i]])
                detected_classes.append(label)


        # Return the image and the first detected class (if any)
        return img, detected_classes[0] if detected_classes else None

    def infer(self, img, track_id):
        # Perform inference and get the annotated image and detected classes with x-coordinates
        annotated_image, detected_classes_with_x = self.infer_image(img)

        # Check if detected_classes_with_x is not None and not empty (optional)
        if detected_classes_with_x:
            # Extract the detected class directly (assuming only one class is present)
            detected_class = detected_classes_with_x

            return {"detected_class": detected_class, "track_id": track_id}

        return {"detected_class": None, "track_id": None}  # Return None if no classes are detected


    def infer_and_save(self, img, track_id):
        # Perform inference and get the annotated image and detected classes with x-coordinates
        annotated_image, detected_classes_with_x = self.infer_image(img)

        # Check if detected_classes_with_x is not None and not empty (optional)
        if detected_classes_with_x:
            # Extract the detected class directly (assuming only one class is present)
            detected_class = detected_classes_with_x

            # Generate the output filename based on the detected class
            output_filename = f"{track_id}_{detected_class}.jpg"

            output_folder = os.path.join('detected_vehiclecolor')
            os.makedirs(output_folder, exist_ok=True)

            # Specify the filename and format of the output video
            output_video_path = os.path.join(output_folder, output_filename)

            output_path = output_filename

            # Save the annotated image
            cv2.imwrite(output_path, annotated_image)

            return {"detected_class": detected_class, "track_id": track_id}

        return {"detected_class": None, "track_id": None}  # Return None if no classes are detected
