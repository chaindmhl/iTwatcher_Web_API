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
from tracking.deepsort_tric.helper.traffic_light import update_light, get_current_state, overlay_traffic_light
from tracking.deepsort_tric.helper.recognize_plate import Plate_Recognizer
from tracking.deepsort_tric.helper.read_plate_comb import YOLOv4Inference
from tracking.models import RedLightLog
from collections import Counter, deque
import numpy as np
import tempfile
from PIL import Image
from tracking.deepsort_tric.helper.light_state import get_current_light_state

stop_threads = False

plate_recognizer = Plate_Recognizer()
ocr = YOLOv4Inference()

class RedLight():
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
    
    def _bbox_intersects_line(self, bbox_top_left, bbox_bottom_right, line_pt1, line_pt2):
        # Check if bbox intersects with the line
        x1, y1 = bbox_top_left
        x2, y2 = bbox_bottom_right

        # Line segment representation
        x3, y3 = line_pt1
        x4, y4 = line_pt2

        # Check if the bbox intersects with the line segment
        def on_segment(px, py, qx, qy, rx, ry):
            if (qy <= max(py, ry) and qy >= min(py, ry) and qx <= max(px, rx) and qx >= min(px, rx)):
                return True
            return False

        def orientation(px, py, qx, qy, rx, ry):
            val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
            if val == 0:
                return 0
            elif val > 0:
                return 1
            else:
                return 2

        o1 = orientation(x1, y1, x2, y2, x3, y3)
        o2 = orientation(x1, y1, x2, y2, x4, y4)
        o3 = orientation(x3, y3, x4, y4, x1, y1)
        o4 = orientation(x3, y3, x4, y4, x2, y2)

        if (o1 != o2 and o3 != o4):
            return True

        if (o1 == 0 and on_segment(x1, y1, x3, y3, x2, y2)):
            return True

        if (o2 == 0 and on_segment(x1, y1, x4, y4, x2, y2)):
            return True

        if (o3 == 0 and on_segment(x3, y3, x1, y1, x4, y4)):
            return True

        if (o4 == 0 and on_segment(x3, y3, x2, y2, x4, y4)):
            return True

        return False
    
    def _plate_within_roi(self, bbox, roi_vertices):
        # Calculate the center of the bounding box
        xmin, ymin, xmax, ymax = map(int, bbox)
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        
        # Check if the bbox_center is within the polygonal ROI
        roi_polygon = np.array(roi_vertices, dtype=np.int32)
        is_within_roi = cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0
        
        return is_within_roi

    def _process_frame(self,frame, input_size, infer, encoder, tracker, memory, nms_max_overlap=0.1):
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

        xa = 0
        ya = 945
        xb = 2497
        yb = 629
        line = [(xa, ya), (xb, yb)]
        cv2.line(frame, line[0], line[1],  (0, 255, 0), 3)
        traffic_light_position = (2400, 50)  # Example position (x, y) on the frame
        
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
            

            angle = self._vector_angle(origin_midpoint, origin_previous_midpoint)

            light_state = get_current_light_state()
            overlay_traffic_light(frame, traffic_light_position, light_state)

                                                                                    
            # Calculate bbox points
            bbox_top_left = (int(bbox[0]), int(bbox[1]))
            bbox_bottom_right = (int(bbox[2]), int(bbox[3]))

            # Check if bbox intersects with the line
            if light_state == "red" and (self._bbox_intersects_line(bbox_top_left, bbox_bottom_right, line[0], line[1]) and angle > 0):
                # Mark violation and store track ID
                track.violated_red_light = True
                cv2.line(frame, line[0], line[1], (0, 0, 255), 3)
                cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 0, 255), 2)
                cv2.putText(frame, f"Violated", (bbox_top_left[0], bbox_top_left[1] - 10), 0,
                            1e-3 * frame.shape[0], (0, 0, 255), 2)
                

            elif hasattr(track, 'violated_red_light') and track.violated_red_light:
                # Display violation status even after crossing
                cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 0, 255), 2)
                cv2.putText(frame, f"Violated", (bbox_top_left[0], bbox_top_left[1] - 10), 0,
                            1e-3 * frame.shape[0], (0, 0, 255), 2)
            else:
                # Display normal bounding box and class name
                cv2.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 255, 0), 2)
            


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
                # Update the traffic light state
                update_light()

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