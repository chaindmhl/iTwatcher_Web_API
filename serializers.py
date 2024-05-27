from rest_framework import serializers
import os
from subprocess import run
import shutil
import tempfile

from django.conf import settings
from tracking.models import (
    Video, 
    CountLog,
    VehicleLog,
    DownloadRequest,
)


class VideoSerializers(serializers.ModelSerializer):
    file = serializers.FileField(max_length=500)

    class Meta:
        model = Video
        fields = ["id", "file"]

    
class CountLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = CountLog
        fields = '__all__'
    
class VehicleLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = VehicleLog
        fields = '__all__'

class DownloadRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = DownloadRequest
        fields = '__all__'

class ProcessVideoSerializers(serializers.Serializer):
    #video_path = serializers.CharField(required=False)
    livestream_url = serializers.CharField(required=False)
    video_path = serializers.PrimaryKeyRelatedField(queryset=Video.objects.all(), required=False)

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if data.get('video_path'):
            data['video_name'] = data['video_path'].name
        return data
    
    def validate(self, attrs):
        video_path = attrs.get('video_path')
        livestream_url = attrs.get('livestream_url')

        if not video_path and not livestream_url:
            raise serializers.ValidationError('Either video path or camera URL must be provided.')

        if video_path and livestream_url:
            raise serializers.ValidationError('Only one of video path or camera URL can be provided.')

        if video_path:
            # check if the file path exists
            video_path_str = str(video_path.file.path)
            if not os.path.exists(video_path_str):
                raise serializers.ValidationError(f'Video file path {video_path_str} does not exist.')
            attrs['video'] = video_path_str

        elif livestream_url:
            livestream_url_str = livestream_url
            attrs['video'] = livestream_url_str
        return attrs

    def create(self, validated_data):
        video = validated_data["video"]
        return video

class LPRSerializers(serializers.Serializer):
    camera_url = serializers.CharField(required=False)
    video_path = serializers.PrimaryKeyRelatedField(queryset=Video.objects.all(),required=False)

    def validate(self, attrs):
        video_path = attrs.get('video_path')
        camera_url = attrs.get('camera_url')

        if not video_path and not camera_url:
            raise serializers.ValidationError('Either video path or camera URL must be provided.')

        if video_path and camera_url:
            raise serializers.ValidationError('Only one of video path or camera URL can be provided.')

        if video_path:
            # check if the file path exists
            video_path_str = str(video_path.file.path)
            if not os.path.exists(video_path_str):
                raise serializers.ValidationError(f'Video file path {video_path_str} does not exist.')
            attrs['input_video'] = video_path_str

        if camera_url:
            attrs['input_video'] = camera_url

        return attrs

    def create(self, validated_data):
        INPUT_VIDEO = validated_data["input_video"]
        output_dir = "/home/itwatcher/tricycle_copy/lpr_output"
        with tempfile.TemporaryDirectory() as tmpdir:
            command = ["python", "lpr.py", f"--input_video={validated_data['input_video']}", f"--output_video={os.path.join(tmpdir, 'output.mp4')}"]
            result = run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Error running lpr.py: {result.stderr.strip()}")
            # Copy the output video to the output directory
            output_file = os.path.join(output_dir, "output.mp4")
            shutil.copyfile(os.path.join(tmpdir, "output.mp4"), output_file)
        return output_file
