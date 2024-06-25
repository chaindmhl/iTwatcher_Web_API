from django.conf import settings
from tracking.deepsort_tric.LPR_all import Plate_Recognition
import os, threading, queue


REQUEST_URL = f"http://{settings.HOST}:8000/"


def process_alllpr(video_path=None, livestream_url=None, is_live_stream=False, video_stream=None):

    processed_frames = []
    
    if video_path:
        # Load the video file
        video_file = video_path

        # Create a folder to store the output frames
        output_folder_path = os.path.join(settings.MEDIA_ROOT, 'lpr_all_videos')
        os.makedirs(output_folder_path, exist_ok=True)

        # Specify the filename and format of the output video
        output_video_path = os.path.join(output_folder_path, f"lpr_{os.path.basename(video_path)}")

        # Create an instance of the VehiclesCounting class
        pr = Plate_Recognition(file_counter_log_name='vehicle_count.log',
                              framework='tf',
                              weights='/home/icebox/itwatcher_api/tracking/deepsort_tric/checkpoints/LPD1',
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
        producer_thread = threading.Thread(target=pr.producer)
        consumer_thread = threading.Thread(target=pr.consumer)


        producer_thread.start()
        consumer_thread.start()

        # Retrieve frames from the processed_queue in real-time
        while producer_thread.is_alive() or consumer_thread.is_alive():
            try:
                processed_frame = pr._processedqueue.get(timeout=1)  # Wait for a frame for 1 second
                yield processed_frame  # Yield the processed frame
            except queue.Empty:
                continue

        # Ensure the threads are terminated
        pr.stop()
        
        # Wait for both threads to finish
        producer_thread.join()
        consumer_thread.join()
        print("stop threads")

        # Retrieve any remaining frames in the queue
        while not pr._processedqueue.empty():
            processed_frame = pr._processedqueue.get()
            yield processed_frame
        
    return processed_frames
