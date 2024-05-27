import datetime, math, tempfile, datetime, cv2, os, time, queue
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
from tracking.deepsort_tric.deep_sort.detection import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
import tracking.deepsort_tric.core.utils as utils
from tracking.deepsort_tric.core.config_PD import cfg
from tracking.deepsort_tric.read_plate_comb import YOLOv4Inference
from tracking.deepsort_tric.warp_plate import warp_plate_image
from tracking.models import BlockLog

from collections import Counter, deque
from PIL import Image
import numpy as np

yolo_inference = YOLOv4Inference()
stop_threads = False


class Pedestrian_Blocking():
    def __init__(self, file_counter_log_name, framework='tf', weights='./checkpoints/yolov4-416',
                size=416, tiny=False, model='yolov4', video='./data/videos/test.mp4',
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

    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)


    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))

    def _process_frame(self,frame, input_size, infer, encoder, tracker, memory, saved_track_ids = {}, plate_num_dict = {}, nms_max_overlap=0.1):
        memory = {}
        class_counter = Counter()

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

        # Define the counting rectangle coordinates
        rectangle_top_left = (500, 600)
        rectangle_bottom_right = (1800, 800)

        # Initialize the timer_start_time dictionary
        timer_start_time = {}
        classes_inside_rectangle = set()

        # Update tracks
        for track in tracker.tracks:
            bbox = track.to_tlbr()

            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            class_name = track.get_class()
            midpoint = track.tlbr_midpoint(bbox)

            # Check if the midpoint of the object's bounding box is inside the defined rectangle
            if (rectangle_top_left[0] <= midpoint[0] <= rectangle_bottom_right[0] and
                    rectangle_top_left[1] <= midpoint[1] <= rectangle_bottom_right[1]):

                # Check if the class is not inside or the timer is not started
                if class_name not in classes_inside_rectangle or class_name not in timer_start_time:
                        # If the class is not inside or the timer is not started, reset the count and start the timer
                    class_counter[class_name] = 0
                    classes_inside_rectangle.add(class_name)
                    timer_start_time[class_name] = datetime.datetime.now()

                # Increment the count
                class_counter[class_name] += 1

                # Check if count is 30 or higher
                if class_counter[class_name] >= 30:
                    # Change the bounding box color to red
                    color = (0, 0, 255)  # Red color
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    allowance = 15
                    xmin = max(0, int(xmin - allowance))
                    ymin = max(0, int(ymin - allowance))
                    xmax = min(frame.shape[1] - 1, int(xmax + allowance))
                    ymax = min(frame.shape[0] - 1, int(ymax + allowance))
                    plate_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
                    frame_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    warped_plate = warp_plate_image(plate_img)
                    plate_resized = cv2.resize(plate_img, (2000, 600), interpolation=cv2.INTER_LANCZOS4)
                    prediction = yolo_inference.infer_image_only_thresh(plate_resized)
                    plate_num = "".join(prediction["detected_classes"])
                    image_name = plate_num + ".jpg"

                    current_timestamp = time.time()
                    if track.track_id not in saved_track_ids:
                        prediction = yolo_inference.infer_image_only_thresh(plate_resized)
                        plate_num = "".join(prediction["detected_classes"])
                        image_name = plate_num + ".jpg"

                        # Save plate_num in the dictionary
                        plate_num_dict[track.track_id] = plate_num

                    block_log = BlockLog.objects.create(
                            filename = image_name,
                            video_file = self._video,
                            plate_number = image_name.split('.')[0],
                            violation = 'Blocking the Pedestrian Lane',
                        )

                    # Create temporary files for plate_img and frame
                    plate_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    Image.fromarray(plate_img).save(plate_img_temp.name)
                    plate_img_temp.close()

                    warped_plate_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    Image.fromarray(warped_plate).save(warped_plate_temp.name)
                    warped_plate_temp.close()

                    frame_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    Image.fromarray(frame).save(frame_img_temp.name)
                    frame_img_temp.close()

                    # Save plate_image using ImageField
                    block_log.plate_image.save(image_name, open(plate_img_temp.name, 'rb'))
                    block_log.warped_image.save(image_name, open(warped_plate_temp.name, 'rb'))
                    # Save frame_image using ImageField
                    block_log.frame_image.save(image_name, open(frame_img_temp.name, 'rb'))

                    # Remove temporary files
                    os.unlink(plate_img_temp.name)
                    os.unlink(frame_img_temp.name)

                else:
                    color = (0, 255, 0)  # Green color

                # Draw a rectangle around the counted object with the determined color
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                # Draw count at the top of the bounding box
                text = f"{class_name}: {class_counter[class_name]}"
                cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw stopwatch timer
                elapsed_time = datetime.datetime.now() - timer_start_time[class_name]
                timer_text = f"Timer: {elapsed_time.seconds}s"
                cv2.putText(frame, timer_text, (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                # If the object is outside the rectangle, remove the class from the set and stop the timer
                classes_inside_rectangle.discard(class_name)
                timer_start_time.pop(class_name, None)

                if track.track_id not in memory:
                    memory[track.track_id] = deque(maxlen=2)

                memory[track.track_id].append(midpoint)
                previous_midpoint = memory[track.track_id][0]

                origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
                cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)

        # Draw the counting rectangle outside the loop
        cv2.rectangle(frame, rectangle_top_left, rectangle_bottom_right, (255, 0, 0), 2)

        # Additional processing with the last track if needed
        if tracker.tracks:  # Check if there are any tracks
            last_track = tracker.tracks[-1]
            print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                last_track.track_id, last_track.get_class(), last_track.to_tlbr())
            )

            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)  # WHITE BOX
            cv2.putText(frame, " " + str(), (int(bbox[0]), int(bbox[1])), 0,
                        1.5e-3 * frame.shape[0], (0, 0, 255), 2)

        
            # Draw the counting rectangle
            cv2.rectangle(frame, rectangle_top_left, rectangle_bottom_right, (255, 0, 0), 2)


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
        

    def show_frames(self):
        global stop_threads
        display_started = False
        slow_motion = False
        slow_motion_factor = 1
        
        while not stop_threads:
            try:
                if not display_started:
                    # Check the length of the processed queue
                    if self._processedqueue.qsize() < 5:                      
                        time.sleep(0.1)  # Add a short delay to avoid busy-waiting
                        continue
                    else:
                        display_started = True
                                    
                # Get the processed frame from the queue
                result = self._processedqueue.get(timeout=1)

                # Display the processed frame
                cv2.imshow('Processed Frame', cv2.resize(result, (1000, 600)))

                # Add the calculated delay based on average processing time
                delay = int((self._time) * 1000)  # Convert seconds to milliseconds
                if delay < 1:
                    delay = 1  # Minimum delay of 1 millisecond
                    
                key = cv2.waitKey(delay) & 0xFF
                
                if key == ord('q'):
                    stop_threads = True
                    break
                elif key == ord('s'):
                    slow_motion = True
                    slow_motion_factor = 11  # Set slow-motion factor for 's'
                elif key == ord('r'):
                    slow_motion = True
                    slow_motion_factor =  22 # Set slower-motion factor for 'r'
                elif key == ord('t'):
                    slow_motion = True
                    slow_motion_factor = 33  # Set slowest-motion factor for 't'
                elif key == ord('n'):
                    slow_motion = False  # Normal speed

                # Slow motion effect by duplicating frames
                if slow_motion:
                    for _ in range(slow_motion_factor - 1):
                        cv2.imshow('Processed Frame', cv2.resize(result, (1000, 600)))
                        key = cv2.waitKey(delay) & 0xFF
                        if key == ord('q'):
                            stop_threads = True
                            session.close()
                            break
                        elif key == ord('s'):
                            slow_motion = True
                            slow_motion_factor = 11  # Set slow-motion factor for 's'
                        elif key == ord('r'):
                            slow_motion = True
                            slow_motion_factor = 22  # Set slower-motion factor for 'r'
                        elif key == ord('t'):
                            slow_motion = True
                            slow_motion_factor = 33  # Set slowest-motion factor for 't'
                        elif key == ord('n'):
                            slow_motion = False  # Normal speed
                            break

            except queue.Empty:
                continue

        cv2.destroyAllWindows()

session.close()
