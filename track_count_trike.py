import queue, os, time, cv2, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tracking.deepsort_tric.deep_sort import preprocessing, nn_matching
from tracking.deepsort_tric.deep_sort.detection1 import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
import tracking.deepsort_tric.core.utils as utils
from tracking.deepsort_tric.core.config_tc import cfg
from tracking.models import VehicleLog
from collections import Counter, deque
import numpy as np
from datetime import date
stop_threads = False

class Track_Count():
    def __init__(self, file_counter_log_name, framework='tf', weights='/home/itwatcher/Desktop/Itwatcher/restricted_yolov4_deepsort/checkpoints/yolov4-416',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4',
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

        self.total_counter = 0
        self.hwy_count = 0
        self.msu_count = 0
        self.sm_count = 0
        self.oval_count = 0
        self.class_counts = 0

    def get_total_counter(self):
        return self.total_counter
    
    def get_hwy_count(self):
        return self.hwy_count

    def get_msu_count(self):
        return self.msu_count
    
    def get_sm_count(self):
        return self.sm_count
    
    def get_oval_count(self):
        return self.oval_count
    
    def get_class_counts(self):
        return self.class_counts
    
    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    
    def _distance_point_to_line(self, point, line_start, line_end):
        # Calculate the distance from point to the line defined by line_start and line_end
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = numerator / denominator
        return distance
    
    def _plate_within_roi(self, bbox, roi_vertices):
        # Calculate the center of the bounding box
        xmin, ymin, xmax, ymax = map(int, bbox)
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        
        # Check if the bbox_center is within the polygonal ROI
        roi_polygon = np.array(roi_vertices, dtype=np.int32)
        is_within_roi = cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0
        
        return is_within_roi

    def _process_frame(self,frame, input_size, infer, encoder, tracker, memory, already_counted, already_passed, passed_counter, class_counts, nms_max_overlap=0.1):
        batch_size =1
        frame_size = frame.shape[:2]
                    
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(dtype = np.float32)
        
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
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self._iou,
            score_threshold=self._score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = ["tricycle (TukTuk)", "tricycle (Jeep)", "tricycle (Sikad)", "tricycle (TopDown)"]

        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression                    
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Define lines
        # hwy
        x1, y1 = 0, 600
        x2, y2 = 2400, 500
        line1 = [(x1, y1), (x2, y2)]

        # cetd
        xa, ya = 0, 750
        xb, yb = 500, 1500
        line2 = [(xa, ya), (xb, yb)]

        # cetd2 (considered with cetd for "Going West")
        xa1, ya1 = 1400, 950
        xb1, yb1 = 2480, 1300
        line5 = [(xa1, ya1), (xb1, yb1)]

        # sm
        xc, yc = 2000, 0
        xd, yd = 2100, 1000
        line3 = [(xc, yc), (xd, yd)]

        # oval
        xe, ye = 200, 1200
        xf, yf = 2200, 1000
        line4 = [(xe, ye), (xf, yf)]

        # Draw green lines
        lines = [line1, line2, line3, line4, line5]
        for line in lines:
            cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
        
        #For Intersection
        roi_vertices = [
            (0, 0),      # Top-left
            (frame.shape[1], 0),  # Top-right
            (frame.shape[1], frame.shape[0]),  # Bottom-right
            (0, frame.shape[0])               # Bottom-left
        ]
        
        # Create a list of lines for the corresponding directions.
        lines = [line1, line2, line3, line4, line5]
        for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue

                        bbox = track.to_tlbr()
                        class_name = track.get_class()

                        midpoint = track.tlbr_midpoint(bbox)
                        origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])
                        if track.track_id not in memory:
                                memory[track.track_id] = deque(maxlen=2)

                        memory[track.track_id].append(midpoint)
                        previous_midpoint = memory[track.track_id][0]

                        origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
                        cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)
                        
                        
                        if self._plate_within_roi(bbox, roi_vertices) and track.track_id not in already_counted:   
                            class_counts[class_name] += 1
                            self.total_counter += 1

                            # Set already counted for ID to true.
                            already_counted.append(track.track_id)  
                            
                        # Check for intersections with each line and update counts accordingly.
                        for line, direction in zip(lines, passed_counter.keys()):
                            if self._intersect(midpoint, previous_midpoint, line[0], line[1]):
                                
                                #cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

                                angle = self._vector_angle(origin_midpoint, origin_previous_midpoint)
                    
                                if direction == "Going North" and angle > 0 and track.track_id not in already_passed:
                                    passed_counter[direction] += 1
                                    already_passed.append(track.track_id)
                                    cv2.line(frame, line[0], line[1], (255, 0, 0), 2)  
                                elif direction == "Going West" and angle < 0 and track.track_id not in already_passed:
                                    passed_counter[direction] += 1
                                    already_passed.append(track.track_id)
                                    cv2.line(frame, line[0], line[1], (255, 0, 255), 2) 
                                elif direction == "Going East" and angle < 0 and track.track_id not in already_passed:
                                    passed_counter[direction] += 1
                                    already_passed.append(track.track_id)
                                    cv2.line(frame, line[0], line[1], (0, 0, 0), 2) 
                                elif direction == "Going South" and angle < 0 and track.track_id not in already_passed:
                                    passed_counter[direction] += 1
                                    already_passed.append(track.track_id)
                                    cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
                                elif direction == "Going West" and angle < 0 and track.track_id not in already_passed:
                                    passed_counter[direction] += 1
                                    already_passed.append(track.track_id)
                                    cv2.line(frame, line[0], line[1], (255, 0, 255), 2)
                                                                                                
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)  # WHITE BOX
                        cv2.putText(frame,str(class_name), (int(bbox[0]), int(bbox[1])), 0,
                                        1e-3 * frame.shape[0], (0, 0, 255), 2)
                        
        # Save the count log to the database
        vehicle_logs = []
        for direction, cnt in passed_counter.items():
            if direction == "Going North":
                hwy_count = cnt
            elif direction == "Going West":
                msu_count = cnt
            elif direction == "Going East":
                sm_count = cnt
            elif direction == "Going South":
                oval_count = cnt
    
        current_date = date.today()

        # Check if a log for the current date already exists in the database
        existing_log = VehicleLog.objects.filter(date=current_date).first()

        # If a log for the current date already exists, update it with the new counts
        if existing_log:
            existing_log.total_count = self.total_counter
            existing_log.hwy_count = hwy_count
            existing_log.msu_count = msu_count
            existing_log.sm_count = sm_count
            existing_log.oval_count = oval_count
            existing_log.class_counts = class_counts
            existing_log.save()
        else:
            # Create a new VehicleLog entry with counts for each direction
            vehicle_log = VehicleLog.objects.create(
                date=current_date,
                filename=self._file_counter_log_name,
                total_count=self.total_counter,
                hwy_count=hwy_count,
                msu_count=msu_count,
                sm_count=sm_count,
                oval_count=oval_count,
                class_counts=class_counts,
            )
            vehicle_log.save()


            # This needs to be larger than the number of tracked objects in the frame.
        if len(memory) > 50:
            del memory[list(memory)[0]]

        # Iterate through passed_counter to display counts for each direction
        h = 0.8 * frame.shape[0]
        for direction, cnt in passed_counter.items():
            class_count_str = f"{direction}:{cnt}"
            
            # Display the count for the direction
            cv2.putText(frame, class_count_str, (int(0.02 * frame.shape[1]), int(h)), 0, 1.3e-3 * frame.shape[0], (255, 0, 0), 3)
            h += 0.05 * frame.shape[0]
        

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
        class_counts = Counter()
        already_counted = deque(maxlen=50) 
        already_passed = deque(maxlen=50)
        passed_counter = {
                        "Going North": 0,
                        "Going West": 0,
                        "Going East": 0,
                        "Going South": 0
                    }

        while not stop_threads:
            try:
                frame = self._queue.get(timeout=1)
                
            except queue.Empty:
                continue
            
            start_time = time.time()

            result = self._process_frame(frame, input_size, infer, encoder, tracker, memory, already_counted, already_passed, passed_counter, class_counts)

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
