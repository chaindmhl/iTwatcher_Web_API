from django.conf import settings
from tracking.deepsort_tric.swerving import Detect_Swerving
import datetime, requests, tempfile, os, cv2, threading, queue
from tracking.models import OutputVideo


REQUEST_URL = f"http://{settings.HOST}:8000/"


def process_swerving(video_path=None, livestream_url=None, is_live_stream=False, video_stream=None):
    # Tricycle Detection and Tracking
    if video_path:
        # Load the video file
        video_file = video_path

        # Create a folder to store the output frames
        output_folder_path = os.path.join(settings.MEDIA_ROOT, 'swerving_videos')
        os.makedirs(output_folder_path, exist_ok=True)
        # Specify the filename and format of the output video
        output_video_path = os.path.join(output_folder_path, f"tracked_{os.path.basename(video_path)}")

        # Create an instance of the VehiclesCounting class
        ds = Detect_Swerving(file_counter_log_name='vehicle_count.log',
                              framework='tf',
                              weights='/home/icebox/itwatcher_api/tracking/deepsort_tric/checkpoints/PlateDetection',
                              size=416,
                              tiny=False,
                              model='yolov4',
                              video=video_file,
                              output=output_video_path,
                              output_format='XVID',
                              iou=0.45,
                              score=0.5,
                              dont_show=False,
                              info=False,
                              detection_line=(0.5, 0),
                              frame_queue = queue.Queue(maxsize=1200),
                              processed_queue=queue.Queue(maxsize=100))

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=ds.producer)
        consumer_thread = threading.Thread(target=ds.consumer)
        showframe_thread = threading.Thread(target=ds.show_frames)

        producer_thread.start()
        consumer_thread.start()
        showframe_thread.start()

        # Wait for both threads to finish
        producer_thread.join()
        consumer_thread.join()
        showframe_thread.join()


        # Create an instance of ObjectTrack and save the video file to it
        output_swerve = OutputVideo.objects.create(
            video_swerve=output_video_path,
        )

        # Create a temporary file for the video file
        output_swerve_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        with open(output_video_path, 'rb') as f:
            output_swerve_temp.write(f.read())
        output_swerve_temp.close()

        # Save the video file using FileField
        output_swerve.video_swerve.save(os.path.basename(output_video_path), open(output_swerve_temp.name, 'rb'))

        # Remove the temporary file
        os.unlink(output_swerve_temp.name)

    elif livestream_url:
        # Check if the livestream_url is valid
        response = requests.get(livestream_url, auth=('username', 'password'))
        if response.status_code != 200:
            # Handle invalid url    
            return {'error': 'Invalid livestream url'}

        # Create a VideoCapture object using the livestream url
        video_file = cv2.VideoCapture(livestream_url)

        # get the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # create a unique output video name using the timestamp
        output_video_path = os.path.join(output_folder_path, f'tracked_livestream_{timestamp}.avi')
        os.makedirs(output_folder_path, exist_ok=True)

        # Create an instance of the VehiclesCounting class
        ds = Detect_Swerving(file_counter_log_name='vehicle_count.log',
                              framework='tf',
                              weights='/home/icebox/itwatcher_api/tracking/deepsort_tric/checkpoints/PlateDetection',
                              size=416,
                              tiny=False,
                              model='yolov4',
                              video=video_file,
                              output=output_video_path,
                              output_format='XVID',
                              iou=0.45,
                              score=0.5,
                              dont_show=False,
                              info=False,
                              detection_line=(0.5, 0),
                              frame_queue = queue.Queue(maxsize=1200),
                              processed_queue=queue.Queue(maxsize=100)),
        

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=ds.producer)
        consumer_thread = threading.Thread(target=ds.consumer)
        showframe_thread = threading.Thread(target=ds.show_frames)

        producer_thread.start()
        consumer_thread.start()
        showframe_thread.start()

        # Wait for both threads to finish
        producer_thread.join()
        consumer_thread.join()
        showframe_thread.join()
