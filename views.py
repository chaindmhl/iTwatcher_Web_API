from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from django.views import View
from PIL import Image
from io import BytesIO

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from django.shortcuts import render,redirect, get_object_or_404
from django.views.decorators.http import require_POST
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
from tracking.models import Video, PlateLog, CountLog, ColorLog, VehicleLog, DownloadRequest, SwerveLog, BlockLog
from tracking.process_tc_trike import process_trackcount_trike
from tracking.process_tc_all import process_trackcount_all
from tracking.process_tc_comb import process_trackcount_comb
from tracking.process_lpr_trike import process_lpr_trike
from tracking.process_lpr_all import process_alllpr
from tracking.process_lpr_comb import process_lpd_comb
from tracking.process_color import process_color
from tracking.process_redlight import process_redlight

from tracking.process_blocking import process_blocking
from django.http import StreamingHttpResponse
from rest_framework.views import APIView

# Define your Django Rest Framework view
from rest_framework.response import Response
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend 'Agg'
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import io, os, base64, subprocess, socket, cv2, json
from django.db.models import Count, F
from datetime import timedelta, datetime
from tracking.forms import SignUpForm
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
import time
from rest_framework.decorators import action
from tracking.deepsort_tric.helper.light_state import set_current_light_state, get_current_light_state


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
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}  # Optionally pass hostname and ip_address to the template
        return render(request, 'html_files/lpr.html', context)



class TrackCountView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/track_count.html', context)  

class ColorRecognitionView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/color.html', context)

class VioDetectionView(View):
    
    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/violation.html', context) 

class MyView(LoginRequiredMixin, View):
    login_url = '/login/'  # Set the URL where unauthenticated users are redirected

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
        return render(request, 'html_files/index.html', context)

class UploadView(View):
    
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.all()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        context = {'users': users, 'videos': videos, 'hostname': hostname, 'ip_address': ip_address}
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

class Streaming:
    def stream_processed_frames(self, processed_frames):
        if not processed_frames:
            return StreamingHttpResponse(status=204)

        def generate():
            for frame in processed_frames:
                try:
                    resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
                    success, buffer = cv2.imencode('.jpg', resized_frame)
                    
                    while not success:
                        print("Failed to encode frame, retrying...")
                        success, buffer = cv2.imencode('.jpg', resized_frame)
                        time.sleep(0.01)
                    
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    yield f"data: {frame_base64}\n\n"
                    
                    time.sleep(0.05)  # Control the frame rate (20 FPS)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        response = StreamingHttpResponse(generate(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['Access-Control-Allow-Origin'] = '*'  # Update with specific domain if needed
        return response
    
class ProcessTrikeViewSet(viewsets.ViewSet, Streaming):
    """
    Perform tricycle Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_trackcount_trike(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CatchAllViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Vehicle Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_trackcount_all(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

      
class ColorViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Vehicle Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_color(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
class CombiViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Vehicle Detection (Trike and Vehicle) in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_trackcount_comb(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        

class LPRTrikeViewSet(viewsets.ViewSet, Streaming):
    """
    Perform LPR-trike in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_lpr_trike(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LPRAllViewSet(viewsets.ViewSet, Streaming):
    """
    Perform LPR-all in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_alllpr(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
class LPRCombiViewSet(viewsets.ViewSet, Streaming):
    """
    Perform LPR_comb in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_lpd_comb(video_path=video_path)  # Define this function accordingly
                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RedLightViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Swerving Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_redlight(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def control_traffic_light(self, request):
        color = request.data.get('color')
        print(f"Received request to change light color to: {color}")
        if color in ['red', 'yellow', 'green']:
            set_current_light_state(color)
            print(f"Light color successfully changed to: {get_current_light_state()}")
            return Response({'status': 'success', 'current_light_state': get_current_light_state()}, status=status.HTTP_200_OK)
        else:
            print("Invalid color received")
            return Response({'error': 'Invalid color'}, status=status.HTTP_400_BAD_REQUEST)
        

class BlockingViewSet(viewsets.ViewSet, Streaming):
    """
    Perform Blocking Detection in Videos
    """

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("livestream_url")
            print("URL:", video_path)

            if video_path:
                # Process video_path and return response
                processed_frames = process_blocking(video_path=video_path)  # Define this function accordingly

                return self.stream_processed_frames(processed_frames)
            else:
                return JsonResponse({"error": "Either video or livestream_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            print("Invalid serializer data:", serializer.errors)
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

        # Get the hostname and IP address
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        context = {
            'plate_logs': plate_logs,
            'hostname': hostname,
            'ip_address': ip_address,
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
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        

        context = {
            'plate_log': plate_log,
            'hostname': hostname,
            'ip_address': ip_address,

        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_frame.html', context)
    
class MapView(View):

    def view_camera_map(request):
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_camera_map.html')

@require_POST
def update_plate_number(request):
    data = json.loads(request.body)
    log_id = data.get('log_id')
    plate_number = data.get('plate_number')

    if log_id and plate_number:
        plate_log = get_object_or_404(PlateLog, id=log_id)
        plate_log.plate_number = plate_number
        plate_log.save()
        return HttpResponse(status=200)  # Successful update
    else:
        return HttpResponse(status=400)  # Bad request

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

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(8, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'yellow', 'black', 'cyan','white','green', 'blue', 'violet']  # Add more colors if needed
        
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
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)
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
        script_path = '/media/icebox/2TB/itwatcher_api/tracking/hikvision/media_download.py'
        camera_ip = request.POST.get('camera_ip')  # Adjust to match your form field name
        start_date = request.POST.get('start_date')
        start_time = request.POST.get('start_time', start)
        end_date = request.POST.get('end_date')
        end_time = request.POST.get('end_time', end)
        content_type = request.POST.get('content_type')  # Assuming you have a content_type field
        channel = request.POST.get('channel')

        if start_date:
            start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")
        else:
            # Set default value for start_datetime
            start_datetime = datetime.now()

        if end_date:
            end_datetime = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S")
        else:
            # Set default value for end_datetime (24 hours after start_datetime)
            start_datetime = datetime.now()
            end_datetime = start_datetime + timedelta(hours=24)

        # Check if start_datetime is greater than or equal to end_datetime
        if start_datetime >= end_datetime:
            error_message = "Start Datetime must be before End Datetime."
            return render(request, self.template_name, {'error_message': error_message})

        directory_name = f"{start_date}_{start_time}_{end_date}_{end_time}_{channel}"
        directory_path = os.path.join('/home/icebox/2TB/itwatcher_api/nvr_videos/10.101.60.148', start_date, channel, start_date)
        directory_exists = os.path.exists(directory_path)

        print(f"Directory path: {directory_path}")
        print(f"Directory exists: {directory_exists}")


        # Check if the directory already exists
        if directory_exists:
            # Directory already exists, prompt user for action
            user_response = request.POST.get('user_response', '').lower().strip()
            if user_response != 'y':
                # User chose not to append, return a response or redirect
                error_message = f"Video Files for {start_date} already exists. Download aborted."
                print(f"Error message: {error_message}")
                return render(request, 'html_files/download_success.html', {'error_message': error_message, 'directory_exists': directory_exists})
                    
            
        # Form the command with arguments
        command = f"python3 {script_path} {camera_ip} {start_date} {start_time} {end_date} {end_time} {channel}"

        if content_type:
            command += " -p"

        # Run the script only if the directory doesn't exist
        if not directory_exists:
            # Run the script and wait for its completion
            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            process.communicate()  # Wait for the process to complete

            # Check the return code for success or failure
            if process.returncode == 0:
                print('Download executed successfully.')
            else:
                print(f'Download execution failed. Return code: {process.returncode}')

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


