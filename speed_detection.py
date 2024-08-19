import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Now import other modules and start TensorFlow code
import tracking.deepsort_tric.core.utils as utils
from tensorflow.python.saved_model import tag_constants
from tracking.deepsort_tric.core.config_tc import cfg
from PIL import Image
import cv2
import numpy as np


from tracking.deepsort_tric.helper.read_plate import YOLOv4Inference
from tracking.deepsort_tric.helper.light_state import get_current_light_state
from tracking.deepsort_tric.helper.traffic_light import overlay_traffic_light
from tracking.deepsort_tric.helper.detect_color import Detect_Color
from tracking.deepsort_tric.helper.detect_plate import Detect_Plate
# deep sort imports
from tracking.deepsort_tric.deep_sort import preprocessing, nn_matching
from tracking.deepsort_tric.deep_sort.detection import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
from collections import deque, defaultdict
import math
import tempfile
import time, queue
from tracking.models import ViolationLog

from collections import Counter, deque
from PIL import Image
import numpy as np

yolo_inference = YOLOv4Inference()
detect_color = Detect_Color()
detect_plate = Detect_Plate()
stop_threads = False


class Speed_Detection():
    def __init__(self, file_counter_log_name, framework='tf', weights='./checkpoints/lpd_comb',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4', outputfile=None,
                output=None, output_format='XVID', iou=0.45, score=0.5,
                dont_show=False, info=False,
                detection_line=(0.5,0), frame_queue = queue.Queue(maxsize=100), processed_queue = queue.Queue(maxsize=100), processing_time=0):
    
        self._file_counter_log_name = file_counter_log_name
        self._framework = framework
        self._weights = weights
        self._size = size
        self._tiny = tiny
        self._model = model
        self._video = video
        self._output = output
        self._output_format = output_format
        self._iou = iou
        self._score = score
        self._dont_show = dont_show
        self._info = info
        self._detect_line_position = detection_line[0]
        self._detect_line_angle = detection_line[1]
        self._queue = frame_queue
        self._processedqueue = processed_queue
        self._time = processing_time
        self._stop_threads = False

    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)


    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    
    def _bbox_within_roi(self, bbox, roi_vertices):
        # Extract the bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, bbox)

        # Define the corners and center points of the bounding box
        points_to_check = [
            (xmin, ymin),          # Top-left corner
            (xmax, ymin),          # Top-right corner
            (xmin, ymax),          # Bottom-left corner
            (xmax, ymax),          # Bottom-right corner
            ((xmin + xmax) // 2, (ymin + ymax) // 2),  # Center point
            ((xmin + xmax) // 2, ymin),                # Top-center
            ((xmin + xmax) // 2, ymax),                # Bottom-center
            (xmin, (ymin + ymax) // 2),                # Left-center
            (xmax, (ymin + ymax) // 2)                 # Right-center
        ]

        # Convert ROI vertices to a NumPy array
        roi_polygon = np.array(roi_vertices, dtype=np.int32)

        # Check if any of the points are within the polygonal ROI
        for point in points_to_check:
            if cv2.pointPolygonTest(roi_polygon, point, False) >= 0:
                return True

        return False

    def _calculate_speed_and_update( self, track, start_time, class_name, high_speed_tracks, line, already_counted, memory, distance=6.5):
        track_id = str(track.track_id)
        elapsed_time = time.time() - start_time
        if already_counted.count(track_id) == 0:
            already_counted.append(track_id)
            speed_ms = distance / elapsed_time
            speed_kh = speed_ms * 3.6
            track.speed = speed_kh
            track.speed_time = time.time()
            # if track.speed > 10:
            #     high_speed_tracks[track_id] = True

    def _process_frame(self,frame, input_size, infer, encoder, tracker, memory = {}, up = {}, down = {}, high_speed_tracks = {}, plate_num_dict = {}, plate_nums = {}, violation_logged = {}, vehicle_colors = {}, already_counted = deque(maxlen=50), nms_max_overlap=0.1):
        batch_size = 1
        frame_size = frame.shape[:2]

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(dtype=np.float32)

        # Repeat along the batch dimension to create a batch of desired size
        batch_data = np.repeat(image_data, batch_size, axis=0)

        # Convert to TensorFlow constant
        batch_data = tf.constant(batch_data, dtype=tf.float32)
        pred_bbox = infer(batch_data)
        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self._iou,
            score_threshold=self._score
        )

        # Convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # Format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # Store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # Read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # By default allow all classes in .names file
        allowed_classes = list(class_names.values())

        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        # Delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # Encode YOLO detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # Run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

                # Call the tracker
        tracker.predict()
        tracker.update(detections)

        x1 = 1
        y1 = 658
        x2 = 1520
        y2 = 522
        line1 = [(x1,y1),(x2,y2)]


        xa = 0
        ya = 812
        xb = 1918
        yb = 608
        line2 = [(xa,ya),(xb,yb)]

        # draw the lines
        # cv2.line(frame, line1[0], line1[1], (0, 255, 0), 3)
        # cv2.line(frame, line2[0], line2[1], (255, 0, 0), 3)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            midpoint = track.tlbr_midpoint(bbox)
            origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

            track_id = str(track.track_id)

            if track.track_id not in memory:
                memory[track.track_id] = deque(maxlen=2)

            memory[track.track_id].append(midpoint)
            previous_midpoint = memory[track.track_id][0]

            origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
            # cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)

            if not hasattr(track, 'speed'):
                track.speed = None
                track.speed_time = None

            # Going up
            if self._intersect(midpoint, previous_midpoint, line2[0], line2[1]) and track_id not in already_counted:
                up[track_id] = time.time()
            if track_id in up:
                if self._intersect(midpoint, previous_midpoint, line1[0], line1[1]) and track_id not in already_counted:
                    self._calculate_speed_and_update(track, up[track_id], class_name, high_speed_tracks, line1, already_counted, memory)

            # Going down
            if self._intersect(midpoint, previous_midpoint, line1[0], line1[1]) and track_id not in already_counted:
                down[track_id] = time.time()
            if track_id in down:
                if self._intersect(midpoint, previous_midpoint, line2[0], line2[1]) and track_id not in already_counted:
                    self._calculate_speed_and_update(track, down[track_id], class_name, high_speed_tracks, line2, already_counted, memory)

            # Display speed if it was calculated recently (within 5 seconds)
            if track.speed and track.speed_time and (time.time() - track.speed_time) < 5:
                # Draw bounding box around the vehicle
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                
                # Initialize to avoid UnboundLocalError
                clr = vehicle_colors.get(track.track_id, "Unknown")
                plate_disp = plate_nums.get(track.track_id, "Unknown")

                # Display "Overspeeding" if speed exceeds 20 Km/h
                if track.speed > 20:
                    cv2.putText(frame, f"Overspeeding: {int(track.speed)} Km/h", 
                                (int(bbox[0]), int(bbox[1]) - 10), 0, 
                                0.7e-3 * frame.shape[0], (0, 255, 0), 2)

                    # Highlight in red if speed is greater than 20 Km/h
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                    # Display plate number and class name above the bounding box
                
                    cv2.putText(frame, f"Vehicle Type:{class_name}", (int(bbox[0]), int(bbox[1]) - 85), 0,
                            0.7e-3 * frame.shape[0], (255, 125, 125), 2)
                    cv2.putText(frame, f"Color:{clr}", (int(bbox[0]), int(bbox[1])- 60), 0,
                                0.7e-3 * frame.shape[0], (255, 125, 125), 2)
                    cv2.putText(frame, f"Plate Number: {plate_disp}", (int(bbox[0]), int(bbox[1])- 35), 0,
                                0.7e-3 * frame.shape[0], (255, 125, 125), 2)
                else:
                    # Just display the speed without "Overspeeding"
                    cv2.putText(frame, f"Speed: {int(track.speed)} Km/h", 
                                (int(bbox[0]), int(bbox[1]) - 10), 0, 
                                0.7e-3 * frame.shape[0], (0, 255, 0), 2)


            if track_id not in violation_logged:
                violation_logged[track_id] = False
            
            if self._intersect(midpoint, previous_midpoint, line1[0], line1[1]) and not violation_logged.get(track_id, False) and track.speed is not None and track.speed > 20:
                try:
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    allowance = 0
                    xmin = max(0, int(xmin - allowance))
                    ymin = max(0, int(ymin - allowance))
                    xmax = min(frame.shape[1] - 1, int(xmax + allowance))
                    ymax = min(frame.shape[0] - 1, int(ymax + allowance))
                
                    
                    vehicle_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    veh_img = cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR)
                    frame_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    veh_resized = cv2.resize(veh_img, (600, 300), interpolation=cv2.INTER_LANCZOS4)
                    veh_name = f"{track.track_id}.jpg"

                    color_pred = detect_color.infer(vehicle_img, track.track_id)
                    clr = color_pred.get("detected_class", "Unknown")  # Default to "Unknown" if not detected
                    if clr is not None:
                        # Store detected color for the track ID
                        vehicle_colors[track.track_id] = clr

                    plate_pred = detect_plate.infer_image(veh_resized)  # change for LPD
                    plate_disp = plate_pred.get("detected_class", "Unknown")  # Default to "Unknown" if not detected
                    cropped_plate = plate_pred.get("cropped_plate", np.zeros((1, 1, 3), dtype=np.uint8))  # Default to an empty image
                    plate_resized = cv2.resize(cropped_plate, (2000, 600), interpolation=cv2.INTER_LANCZOS4)
                    pred = yolo_inference.infer_image_only_thresh(plate_resized)
                    plate_disp = "".join(pred.get("detected_classes", ["Unknown"]))  # Default to "Unknown" if not detected
                    if plate_disp is not None:
                        # Store detected color for the track ID
                        plate_nums[track.track_id] = plate_disp

                    image_name = plate_disp + ".jpg"

                    # Save to the database
                    current_timestamp = time.time()
                    if plate_disp not in plate_num_dict:
                        plate_num_dict[plate_disp] = current_timestamp

                    # Save the plate log to the database
                    vio_log = ViolationLog.objects.create(
                        video_file=self._video,
                        vehicle_type = class_name,
                        violation='Overspeeding',
                        plate_number=plate_disp,
                        vehicle_color = clr,           
                    )

                    vehicle_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    Image.fromarray(cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR)).save(vehicle_img_temp.name)
                    vehicle_img_temp.close()   

                    # Create temporary files for plate_img and frame
                    plate_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    Image.fromarray(cropped_plate).save(plate_img_temp.name)
                    plate_img_temp.close()

                    frame_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    Image.fromarray(frame_img).save(frame_img_temp.name)
                    frame_img_temp.close()

                    vio_log.vehicle_image.save(veh_name, open(vehicle_img_temp.name, 'rb'))
                    # Save plate_image using ImageField
                    vio_log.plate_image.save(image_name, open(plate_img_temp.name, 'rb'))
                    # Save frame_image using ImageField
                    vio_log.frame_image.save(image_name, open(frame_img_temp.name, 'rb'))

                    # Remove temporary files
                    os.unlink(plate_img_temp.name)
                    os.unlink(frame_img_temp.name)
                
                except cv2.error as e:
                    continue


        # This needs to be larger than the number of tracked objects in the frame.
        if len(memory) > 50:
            del memory[list(memory)[0]]

        result = np.asarray(frame)
        return result

    
    def producer(self):

        global stop_threads
        frame_count = 0
        skip_frames = 1

        cap = cv2.VideoCapture(self._video)
        if not cap.isOpened():
            # print("Error: Unable to open the video stream.")
            return
        
        while not stop_threads:
            ret, frame = cap.read()
            # print("reading video...")
            if not ret:
                # print("Failed to retrieve frame. Pausing...")
                stop_threads = False
                continue
            frame_count +=1

            if frame_count % skip_frames == 0: 
                try:
                    self._queue.put(frame, timeout=1)

                except queue.Full:
                    
                    time.sleep(1)
                    continue
                    

        cap.release()

    def consumer(self):
        global stop_threads
        input_size = self._size
        total_processing_time = 0
        num_frames_processed = 0

        # Load configuration for object detector
        saved_model_loaded = tf.saved_model.load(self._weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        model_filename = '/home/icebox/itwatcher_api/tracking/deepsort_tric/model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        max_cosine_distance = 0.4
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        memory = {}

        while not stop_threads:
            try:
                frame = self._queue.get(timeout=1)
                
            except queue.Empty:
                continue
            
            start_time = time.time()

            result = self._process_frame(frame, input_size, infer, encoder, tracker, memory)

            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            num_frames_processed += 1

            if result is not None and len(result) > 0:
                self._processedqueue.put(result)

        # Calculate average processing time
        average_processing_time = 0
        if num_frames_processed > 0:
            average_processing_time = total_processing_time / num_frames_processed
            print(f"Average processing time: {average_processing_time:.3f} seconds") 

        self._time = average_processing_time  # Set the attribute
        print(self._time)
        return average_processing_time  # Return average processing time
        
    def retrieve_processed_frames(self):
            processed_frames = []
            while not self._processedqueue.empty():
                processed_frames.append(self._processedqueue.get())
            return processed_frames
        
    def stop(self):
        self._stop_threads = True
    
session.close()
