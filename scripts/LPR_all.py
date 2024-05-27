import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import tracking.deepsort_tric.core.utils as utils
from tracking.deepsort_tric.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tracking.deepsort_tric.core.config_lprall import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tracking.models import PlateLog
from django.core.files.base import ContentFile

# deep sort imports
from tracking.deepsort_tric.deep_sort import preprocessing, nn_matching
from tracking.deepsort_tric.deep_sort.detection import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
import datetime
from collections import Counter, deque
import math
from darknet.read_plate_all import YOLOv4Inference
import tempfile


class Plate_Recognition():
    def __init__(self, file_counter_log_name, framework='tf', weights='/home/icebox/itwatcher_api/tracking/deepsort_tric/checkpoints/lpr_all',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4',
                output=None, output_format='XVID', iou=0.45, score=0.5,
                dont_show=False, info=False,
                detection_line=(0.5,0)):
        
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
      
    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)


    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    
    def _plate_within_roi(self, bbox, roi_vertices):
        # Calculate the center of the bounding box
        xmin, ymin, xmax, ymax = map(int, bbox)
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        
        # Check if the bbox_center is within the polygonal ROI
        roi_polygon = np.array(roi_vertices, dtype=np.int32)
        is_within_roi = cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0
        
        return is_within_roi


    def run(self):
            # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        show_detections = False

        # initialize deep sort
        model_filename = '/home/icebox/itwatcher_api/tracking/deepsort_tric/model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = self._size
        video_path = self._video

        # load tflite model if flag is set
        if self._framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self._weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(self._weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        # get video ready to save locally if flag is set
        if self._output:
            # by default VideoCapture returns float instead of intclass_counter
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self._output_format)
            out = cv2.VideoWriter(self._output, codec, fps, (width, height))

        frame_num = 0
        current_date = datetime.datetime.now().date()
        count_dict = {}  # initiate dict for storing counts

        total_counter = 0
        up_count = 0
        down_count = 0

        class_counter = Counter()  # store counts of each detected class
        already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
        intersect_info = []  # initialise intersection list

        memory = {}
        skip_frames = 1
        processed_frame = 0
        total_frames = 0
        total_processing_time = 0
        total_delay = 0
        yolo_inference = YOLOv4Inference()
        
        while True:
            return_value, frame = vid.read()  
            #return_value = vid.grab()
            if return_value:
                total_frames += 1

                if total_frames % skip_frames == 0:
                    #_,frame = vid.retrieve()
                    #frame = cv2.UMat(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    processed_frame +=1
                    frame_num +=1
                    print('Frame #: ', frame_num)
                    frame_size = frame.shape[:2]
                    
                    image_data = cv2.resize(frame, (input_size, input_size))
                    image_data = image_data / 255.

                    image_data = image_data[np.newaxis, ...].astype(np.float32)
                    start_time = time.time()

                    # run detections on tflite if flag is set
                    if self._framework == 'tflite':
                        interpreter.set_tensor(input_details[0]['index'], image_data)
                        interpreter.invoke()
                        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                        # run detections using yolov3 if flag is set
                        if self._model == 'yolov3' and self._tiny == True:
                            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                            input_shape=tf.constant([input_size, input_size]))
                        else:
                            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                            input_shape=tf.constant([input_size, input_size]))
                    else:
                        batch_data = tf.constant(image_data)
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
                    #allowed_classes = list(class_names.values())
                    allowed_classes = ['License_Plate']

                    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                    names = []
                    deleted_indx = []
                    
                    for i in range(num_objects):
                        class_indx = int(classes[i])
                        class_name = class_names[class_indx]
                        if class_name not in allowed_classes:
                            deleted_indx.append(i)
                        else:
                            names.append(class_name)
                            
                    names = np.array(names)
                    count = len(names)
                    '''if count:
                        cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                        print("Objects being tracked: {}".format(count))'''
                    
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

                    yp = math.tan(self._detect_line_angle*math.pi/180) * frame.shape[1] / 2
                    x1 = 0
                    y1 = int(frame.shape[0] / 2) + 100 #int(self._detect_line_position * frame.shape[0] + yp)
                    x2 = int(frame.shape[1])
                    y2 = 500#int(self._detect_line_position * frame.shape[0] + yp)

                    line = [(x1, y1), (x2, y2)]

                    # draw yellow line
                    #cv2.line(frame, line[0], line[1], (200, 200, 200), 1)

                    # Create a dictionary to keep track of the already saved track IDs
                    saved_track_ids = {}
                    
                    # Define the vertices of the polygon (clockwise or counterclockwise order)
                    roi_vertices = [
                        (0, int(frame.shape[0] / 2)-100),      # Top-left
                        (frame.shape[1], 0),  # Top-right
                        (frame.shape[1], int(frame.shape[0])),  # Bottom-right
                        (0, frame.shape[0])               # Bottom-left
                    ]

                    # Convert the vertices to a NumPy array of shape (vertices_count, 1, 2)
                    roi_vertices_np = np.array(roi_vertices, dtype=np.int32)
                    roi_vertices_np = roi_vertices_np.reshape((-1, 1, 2))

                    # Draw the polygonal ROI using polylines
                    cv2.polylines(frame, [roi_vertices_np], isClosed=True, color=(0, 255, 0), thickness=2)
                    plate_num_dict = {}
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
                        #cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 1)
                        # Assign the track_id outside of the intersect block
                        track_id = str(track.track_id)

                        # Check if the object is within the ROI
                        if self._plate_within_roi(bbox, roi_vertices):
                            print("Object is within ROI:", track_id)
                        
                            xmin, ymin, xmax, ymax = map(int, bbox)
                            plate_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                            plate_resized = cv2.resize(plate_img, (2000, 600), interpolation=cv2.INTER_LANCZOS4)
                            prediction = yolo_inference.infer_and_save(plate_resized)
                            plate_num = "".join(prediction["detected_classes"])   

                            # Display the plate number on the frame
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                            cv2.putText(frame, plate_num, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            1e-3 * frame.shape[0], (0, 255, 0), 2)
                                            
                        if self._intersect(midpoint, previous_midpoint, line[0], line[1]) and track.track_id not in already_counted:
                                print("Entering ROI:", track.track_id)
                                class_counter[class_name] += 1
                                total_counter += 1

                                # Set already counted for ID to true.
                                already_counted.append(track.track_id)  

                                intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
                                angle = self._vector_angle(origin_midpoint, origin_previous_midpoint)
                                intersect_info.append([class_name, origin_midpoint, angle, intersection_time])

                                if angle > 0:
                                    up_count += 1
                                if angle < 0:
                                    down_count += 1
                                
                                
                                xmin, ymin, xmax, ymax = map(int, bbox)
                                plate_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                                plate_resized = cv2.resize(plate_img, (2000,600), interpolation = cv2.INTER_LANCZOS4)
                                # Save the cropped image only once for each track ID
                                #track_id = str(track.track_id)
                                #if track_id not in saved_track_ids and self._intersect(midpoint, previous_midpoint, line[0], line[1]):
                                if track_id not in saved_track_ids:
                                    saved_track_ids[track_id] = True
                                    
                                    prediction = yolo_inference.infer_and_save(plate_resized)
                                    plate_num = "".join(prediction["detected_classes"])
                                    image_name = plate_num + ".jpg"

                                    # Save plate_num in the dictionary
                                    plate_num_dict[track_id] = plate_num


                            # Save the count log to the database
                                plate_log = PlateLog.objects.create(
                                    filename = image_name,
                                    plate_number = image_name.split('.')[0],
                                )
                            
                                # Create temporary files for plate_img and frame
                                plate_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                                Image.fromarray(plate_img).save(plate_img_temp.name)
                                plate_img_temp.close()

                                frame_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                                Image.fromarray(frame).save(frame_img_temp.name)
                                frame_img_temp.close()

                                # Save plate_image using ImageField
                                plate_log.plate_image.save(image_name, open(plate_img_temp.name, 'rb'))

                                # Save frame_image using ImageField
                                plate_log.frame_image.save(image_name, open(frame_img_temp.name, 'rb'))

                                # Remove temporary files
                                os.unlink(plate_img_temp.name)
                                os.unlink(frame_img_temp.name)

                        #result = np.asarray(frame)
                        #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                       
                    # if enable info flag then print details about each track
                        if self._info:
                            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

                # Delete memory of old tracks.
                        # This needs to be larger than the number of tracked objects in the frame.
                    if len(memory) > 50:
                        del memory[list(memory)[0]]

                    if show_detections:
                        for det in detections:
                            bbox = det.to_tlbr()
                            score = "%.2f" % (det.confidence * 100) + "%"
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)  # BLUE BOX
                            if len(classes) > 0:
                                det_cls = det.cls
                                cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                                            1.5e-3 * frame.shape[0], (0, 255, 0), 2)

                    end_time = time.time()
                    processing_time = (end_time - start_time)*1000
                    total_processing_time += processing_time
                    delay = (time.time() - start_time)*1000
                    total_delay += delay

                    # calculate frames per second of running detections
                    fps = 1.0 / (time.time() - start_time)
                    #print("FPS: %.2f" % fps)
                    result = frame.copy()
                    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if self._dont_show:
                        cv2.imshow(self._file_counter_log_name, result)

                    # if output flag is set, save video file
                    if self._output:
                        out.write(result)
            else:
                print('Video has ended or failed, try a different video format!')
                break

        average_processing_time = total_processing_time / total_frames
        average_delay = total_delay / total_frames

        print('Average Processing Time: ', average_processing_time,'ms')
        print('Average Processing Delay: ', average_delay,'ms')  
        
        vid.release()
        cv2.destroyAllWindows()
