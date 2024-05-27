from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from django.views import View
from PIL import Image
from io import BytesIO
from django.conf import settings

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from django.shortcuts import render,redirect
from django.http import FileResponse
from rest_framework import viewsets, status, generics
from rest_framework.response import Response
from tracking.serializers import (
    VideoSerializers,
    ProcessVideoSerializers,
    LPRSerializers,
    CountLogSerializer,
    VehicleLogSerializer,
    DownloadRequestSerializer,
)
from tracking.models import Video, PlateLog, CountLog, ColorLog, VehicleLog, DownloadRequest, SwerveLog, BlockLog, NVRVideo
from tracking.process_tc_trike import process_trackcount_trike
from tracking.process_tc_all import process_trackcount_all
from tracking.process_tc_comb import process_trackcount_comb
from tracking.process_lpr_trike import process_lpr_trike
from tracking.process_lpr_all import process_alllpr
from tracking.process_lpr_comb import process_lpd_comb
from tracking.process_color import process_color
from tracking.process_swerving import process_swerving
from tracking.process_blocking import process_blocking
from rest_framework.decorators import action
#from tracking.process_frontside import process_frontside

# Define your Django Rest Framework view
from rest_framework.response import Response
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import io, os, tempfile, base64, subprocess
from django.db.models import Count, F
from datetime import timedelta, datetime
from django.utils import timezone
from tracking.forms import SignUpForm
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404
from .models import NVRVideo
from django.conf import settings

# Create your views here.
class DarknetTrainView(View):
    def post(self, request, *args, **kwargs):
        # Get parameters from the form
        data_path = request.POST.get('data_path')
        cfg_path = request.POST.get('cfg_path')
        weight_path = request.POST.get('weight_path')
        # ... add other parameters as needed

        # Set the working directory to the Darknet folder
        darknet_path = '/home/icebox/darknet'  # Replace with the actual path to your Darknet folder
        os.chdir(darknet_path)

        # Run Darknet training command
        command = f'./darknet detector train {data_path} {cfg_path} {weight_path}'

        try:
            result = subprocess.check_output(command, shell=True)
            return HttpResponse(result)
        except subprocess.CalledProcessError as e:
            return HttpResponse(str(e))

    def get(self, request, *args, **kwargs):
        return render(request, 'html_files/train.html')

class CustomLoginView(AuthLoginView):
    template_name = 'html_files/login.html'

class SignupView(View):

    def get(self, request):
        form = SignUpForm()
        return render(request, 'html_files/signup.html', {'form': form})

    def post(self, request):
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to the login page after successful signup
        return render(request, 'html_files/signup.html', {'form': form})

class LPRView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        context = {'users': users, 'videos': videos}
        return render(request, 'html_files/lpr.html', context)

class TrackCountView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        context = {'users': users, 'videos': videos}
        return render(request, 'html_files/track_count.html', context)  

class ColorRecognitionView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        context = {'users': users, 'videos': videos}
        return render(request, 'html_files/color.html', context)

class VioDetectionView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        context = {'users': users, 'videos': videos}
        return render(request, 'html_files/violation.html', context) 

class MyView(LoginRequiredMixin, View):
    login_url = '/login/'  # Set the URL where unauthenticated users are redirected

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        context = {'users': users, 'videos': videos}
        return render(request, 'html_files/index.html', context)

class UploadView(View):
    
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        context = {'users': users, 'videos': videos}
        return render(request, 'html_files/upload_video.html', context)

class CountLogViewSet(viewsets.ModelViewSet):
    queryset = CountLog.objects.all()
    serializer_class = CountLogSerializer

class VehicleLogViewSet(viewsets.ModelViewSet):
    queryset = VehicleLog.objects.all()
    serializer_class = VehicleLogSerializer
       
# Create your views here.
class VideoUploadViewSet(viewsets.ModelViewSet):
    """
    Uploads a File
    """

    queryset = Video.objects.all()
    serializer_class = VideoSerializers
    permission_classes = [IsAuthenticated]


class ProcessTrikeViewSet(viewsets.ViewSet):
    """
    Perform tricycle Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_trackcount_trike(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_trackcount_trike(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CatchAllViewSet(viewsets.ViewSet):
    """
    Perform Vehicle Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_trackcount_all(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_trackcount_all(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class ColorViewSet(viewsets.ViewSet):
    """
    Perform Vehicle Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_color(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_color(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class CombiViewSet(viewsets.ViewSet):
    """
    Perform Vehicle Detection (Trike and Vehicle) in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_trackcount_comb(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_trackcount_comb(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
       
    
class LPRTrikeViewSet(viewsets.ViewSet):
    """
    Perform LPR-trike in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_lpr_trike(video_path=video_path)   
            elif livestream_url:
                stream_path = livestream_url
                context = process_lpr_trike(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LPRAllViewSet(viewsets.ViewSet):
    """
    Perform LPR-all in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_alllpr(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_alllpr(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  

class LPRCombiViewSet(viewsets.ViewSet):
    """
    Perform LPR_comb in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_lpd_comb(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_lpd_comb(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 

class SwervingViewSet(viewsets.ViewSet):
    """
    Perform Swerving Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_swerving(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_swerving(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class BlockingViewSet(viewsets.ViewSet):
    """
    Perform Blocking Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_blocking(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_blocking(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

         
class ColorView(View):

    def get(self, request):
        color_logs = ColorLog.objects.all()
        context = {
            'color_logs': color_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/display_color.html', context)
    def post(self, request):
        # Handle POST requests if needed
        pass


class PlateView(View):
    def get(self, request):
        # Select distinct filenames with timestamps within a 5-minute range and having three or more characters
        distinct_filenames = PlateLog.objects.filter(
            timestamp__range=(F('timestamp') - timedelta(minutes=5), F('timestamp') + timedelta(minutes=5)),
            filename__in=PlateLog.objects.values('filename').annotate(count=Count('filename')).filter(count__gte=3).values('filename')
        ).values('filename').distinct()

        # Create a list of IDs to exclude
        exclude_ids = PlateLog.objects.exclude(
            filename__in=distinct_filenames
        ).values('id')

        # Exclude the unwanted entries
        PlateLog.objects.exclude(id__in=exclude_ids).delete()

        # Retrieve the remaining PlateLog records
        plate_logs = PlateLog.objects.all()

        context = {
            'plate_logs': plate_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/display_plates.html', context)

    def post(self, request):
        # Handle POST requests if needed
        pass

class FrameColorView(View):

    def view_colorframe(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        color_log = ColorLog.objects.get(id=log_id)
        context = {
            'color_log': color_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_colorframe.html', context)
    
class FrameSwerveView(View):

    def view_swerveframe(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        swerve_log = SwerveLog.objects.get(id=log_id)
        context = {
            'swerve_log': swerve_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_swerveframe.html', context)
    
class FrameBlockView(View):

    def view_blockframe(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        block_log = BlockLog.objects.get(id=log_id)
        context = {
            'block_log': block_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_blockframe.html', context)

class SwerveView(View):
    def get(self, request):
        swerve_logs = SwerveLog.objects.all()
        context = {
            'swerve_logs': swerve_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/swerving_list.html', context)
    def post(self, request):
        # Handle POST requests if needed
        pass

class BlockView(View):
    def get(self, request):
        block_logs = BlockLog.objects.all()
        context = {
            'block_logs': block_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/blocking_list.html', context)
    def post(self, request):
        # Handle POST requests if needed
        pass

class FrameView(View):

    def view_frame(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        plate_log = PlateLog.objects.get(id=log_id)
        context = {
            'plate_log': plate_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_frame.html', context)
    
class MapView(View):

    def view_camera_map(request, log_id):
        plate_log = PlateLog.objects.get(id=log_id)
        context = {
            'plate_log': plate_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_camera_map.html', context)

class CountLogListView(View):

    template_name = 'html_files/count_log_list.html'

    def get(self, request, *args, **kwargs):
        count_logs = CountLog.objects.all()
        context = {'count_logs': count_logs}
        return render(request, self.template_name, context)
            
class VehicleLogListView(View):

    template_name = 'html_files/vehicle_log_list.html'

    def get(self, request, *args, **kwargs):
        vehicle_logs = VehicleLog.objects.all()
        context = {'vehicle_logs': vehicle_logs}
        return render(request, self.template_name, context)
    
class TrikeVehicleLogListView(View):

    template_name = 'html_files/trikeall_log_list.html'

    def get(self, request, *args, **kwargs):
        trikeall_logs = VehicleLog.objects.all()
        context = {'trikeall_logs': trikeall_logs}
        return render(request, self.template_name, context)

class TricycleCountGraphView(View):
    template_name = 'html_files/tricycle_count_graph.html'  # Path to your template

    def get(self, request, log_id):
        # Retrieve the log entry based on log_id
        log = CountLog.objects.get(id=log_id)  # Adjust this based on your model

        # Extract class names and counts from the log
        class_names = list(log.class_counts.keys())
        class_counts = list(log.class_counts.values())

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(6, 4))
        plt.bar(class_names, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Bar Graph')
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        plt.close(bar_figure)

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)

        # Convert the buffer data to base64 for embedding in the HTML
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()
        line_graph_data = base64.b64encode(line_buffer.getvalue()).decode()

        context = {
            'bar_graph_data': bar_graph_data,
            'pie_graph_data': pie_graph_data,
            'line_graph_data': line_graph_data,
            'log_id': log_id,
        }

        return render(request, self.template_name, context)

class VehicleCountGraphView(View):
    template_name = 'html_files/vehicle_count_graph.html'  # Path to your template

    def get(self, request, log_date, log_id):
        # Retrieve the log entry based on log_id
        log = VehicleLog.objects.get(id=log_id)  # Adjust this based on your model
        # logs = VehicleLog.objects.filter(timestamp__date=log_date)

        # Extract class names and counts from the log
        class_names = list(log.class_counts.keys())
        class_counts = list(log.class_counts.values())
        # Extract vehicle types and counts from the logs
        # class_names = list(set(log.vehicle_type for log in logs))
        # class_counts = [logs.filter(vehicle_type=vehicle_type).count() for vehicle_type in class_names]
        print("Class Names:", class_names)
        print("Class Counts:", class_counts)

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(8, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'yellow', 'black']  # Add more colors if needed
        #plt.bar(class_names, class_counts)
        # Calculate cumulative counts for each vehicle type
        cumulative_counts = [0] * len(class_names)
        for i, count in enumerate(class_counts):
            plt.bar(log_date, count, bottom=cumulative_counts, color=colors[i], label=class_names[i])
            cumulative_counts[i] += count

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Bar Graph for {log_date}')
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        plt.close(bar_figure)

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)

        # Convert the buffer data to base64 for embedding in the HTML
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()
        line_graph_data = base64.b64encode(line_buffer.getvalue()).decode()

        context = {
            'bar_graph_data': bar_graph_data,
            'pie_graph_data': pie_graph_data,
            'line_graph_data': line_graph_data,
            'log_date': log_date,
            
        }

        return render(request, self.template_name, context)

class DownloadRequestListCreateView(generics.ListCreateAPIView):
    template_name = 'html_files/download_video.html'
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

    def post(self, request, *args, **kwargs):
        # Handle form submission and run the media_download.py script
        # Assuming you have form handling logic here...
        start = '00:00:00'
        end = '23:59:59'
        # Run the media_download.py script
        script_path = '/home/icebox/itwatcher_api/tracking/hikvision/media_download.py'
        camera_ip = request.POST.get('camera_ip')  # Adjust to match your form field name
        start_date = request.POST.get('start_date')
        end_date = start_date
        start_time = request.POST.get('start_time', start)
        end_time = end
        channel = request.POST.get('channel')

        # Set default values for start_datetime and end_datetime
        start_datetime = None
        end_datetime = None

        # Check if start_date and start_time are provided
        if start_date and start_time:
            start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")

        # Set end_datetime to start_datetime at 23:59:59
        end_datetime = start_datetime.replace(hour=23, minute=59, second=59)

        directory_name = f"{start_date}_{start_time}_{channel}"
        # Create a folder to store the output frames
        output_folder_path = os.path.join(settings.MEDIA_ROOT, 'nvr_videos')


        directory_path = os.path.join(output_folder_path,'10.101.60.148', start_date, channel, start_date)
        # directory_exists = os.path.exists(directory_path)
        video_file = f"{directory_path}/{start_time}.mp4"
        video_db = f"{directory_path}/{start_time}"
        directory_exists = os.path.exists(video_file)
        print(f"Directory path: {video_file}")
        print(f"Directory exists: {directory_exists}")

        # Check if the directory already exists
        if directory_exists:
            # Directory already exists, prompt user for action
            user_response = request.POST.get('user_response', '').lower().strip()
            if user_response != 'y':
                # User chose not to append, return a response or redirect
                error_message = f"Video Files for {video_file} already exists. Download aborted."
                print(f"Error message: {error_message}")
                return render(request, 'html_files/download_success.html', {'error_message': error_message, 'directory_exists': directory_exists})

        # Form the command with arguments
        command = f"python3 {script_path} {camera_ip} {start_date} {start_time} {end_date} {end_time} {channel}"

        # Run the script only if the directory doesn't exist
        if not directory_exists:
            # Run the script and wait for its completion
            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()  # Capture stdout and stderr
            print(f'Script output: {stdout.decode()}')
            print(f'Script error: {stderr.decode()}')

            # Check the return code for success or failure
            if process.returncode == 0:
                print('Download executed successfully.')

                # Save the DownloadRequest object to the database
                download_request = DownloadRequest(
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    channel=channel,
                    directory_path=directory_path,
                    timestamp=timezone.now(),
                )
                download_request.save()
                output_nvr = NVRVideo.objects.create(
                    video_nvr=video_file,
                )
                # Create a temporary file for the video file
                output_nvr_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                with open(video_db, 'rb') as f:
                    output_nvr_temp.write(f.read())
                output_nvr_temp.close()

                # Save the video file using FileField
                output_nvr.video_nvr.save(os.path.basename(video_db), open(output_nvr_temp.name, 'rb'))

                # Remove the temporary file
                os.unlink(output_nvr_temp.name)
                
                # Return a success response
                return JsonResponse({'message': 'Video files saved to the database'})
            else:
                print(f'Download execution failed. Return code: {process.returncode}')
                # Return an error response
                return JsonResponse({'error': 'Download execution failed'}, status=500)

        # Render the form again (or handle success differently if needed)
        return render(request, 'html_files/download_success.html', {'directory_exists': directory_exists})


def success_page(request):
    return render(request, 'html_files/download_success.html')
           
class DownloadRequestDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer

def generate_report(request, log_id):
    # Fetch the SwerveLog instance
    swerve_log_instance = SwerveLog.objects.get(pk=log_id)

    # Construct the filename with the unique identifier
    filename = f"Violation_Report_{log_id}.pdf"

    # Create a BytesIO buffer to store the PDF
    buffer = BytesIO()

    # Create a canvas
    c = canvas.Canvas(buffer, pagesize=A4)
    # Set left margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, 750, "iTWatcher")
    c.drawString(30, 620, "LICENSE PLATE RECOGNITION")

    # Define the font and font size
    c.setFont("Helvetica", 11)
    c.drawString(30, 730, "Address: H-Building, Mindanao State University - General Santos City, General Santos City")
    c.drawString(30, 710, "Contact No.: 09171474280")
    c.drawString(30, 690, "Email: inteltraf.watcher@msugensan.edu.ph")
    
    # Write the table content
    c.drawString(30, 570, swerve_log_instance.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    c.drawString(150, 570, str(swerve_log_instance.video_file))
    c.drawString(260, 570, "Laurel Avenue")
    c.drawString(360, 570, str(swerve_log_instance.plate_number))
    c.drawString(460, 570, str(swerve_log_instance.violation))
    
    c.setFont("Helvetica-Bold", 12)

    # Write the table headers
    c.drawString(30, 590, "Date and Time")
    c.drawString(150, 590, "Video Source")
    c.drawString(260, 590, "Location")
    c.drawString(360, 590, "Plate Number")
    c.drawString(460, 590, "Traffic Violation")
    c.drawString(30, 520,"License Plate Image:")
    c.drawString(30, 420,"Warped License Plate Image:")
    c.drawString(30, 320,"Screen Capture:")
    # Calculate the position and size of the plate image
    plate_image_width = 100
    plate_image_height = 50
    plate_image_x = 100
    plate_image_y = 500 - plate_image_height

    # Draw the plate image if available
    if swerve_log_instance.plate_image:
        plate_image_path = swerve_log_instance.plate_image.path
        plate_image = Image.open(plate_image_path)
        c.drawImage(plate_image_path, plate_image_x, plate_image_y, width=plate_image_width, height=plate_image_height)

    # Calculate the position and size of the plate image
    warped_image_width = 100
    warped_image_height = 50
    warped_image_x = 100
    warped_image_y = 400 - warped_image_height

    # Draw the plate image if available
    if swerve_log_instance.warped_image:
        warped_image_path = swerve_log_instance.warped_image.path
        warped_image = Image.open(warped_image_path)
        c.drawImage(warped_image_path, warped_image_x, warped_image_y, width=warped_image_width, height=warped_image_height)

    # Calculate the position and size of the frame image
    frame_image_width = 300
    frame_image_height = 200
    frame_image_x = 100
    frame_image_y = 300 - frame_image_height

    # Draw the frame image if available
    if swerve_log_instance.frame_image:
        frame_image_path = swerve_log_instance.frame_image.path
        frame_image = Image.open(frame_image_path)
        c.drawImage(frame_image_path, frame_image_x, frame_image_y, width=frame_image_width, height=frame_image_height)

    # Save the PDF
    c.showPage()
    c.save()

    # Go to the beginning of the buffer
    buffer.seek(0)

    # Create a Django response and return the PDF with the unique filename
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.write(buffer.getvalue())
    return response