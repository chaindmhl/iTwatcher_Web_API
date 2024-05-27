from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.html import format_html
from PIL import Image
from datetime import datetime
import os
from django.core.files import File



# Create your models here.
class Video(models.Model):
    file = models.FileField(upload_to="videos/", blank=True)

class OutputVideo(models.Model):
    video_track = models.FileField(upload_to='tracked_videos')

class LPRVideo(models.Model):
    video_lpr = models.FileField(upload_to='LPR_videos')

class ColorVideo(models.Model):
    video_color = models.FileField(upload_to='color_videos')

class SwervingVideo(models.Model):
    video_swerve = models.FileField(upload_to='swerving_videos')

class BlockingVideo(models.Model):
    video_block = models.FileField(upload_to='blocking_videos')

class NVRVideo(models.Model):
    video_nvr = models.FileField(upload_to='nvr_videos', default='nvr_videos/video.mp4')


class BasenameField(models.CharField):
    def to_python(self, value):
        value = super().to_python(value)
        if value is not None:
            value = os.path.basename(value)
        return value

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

class DownloadRequest(models.Model):
    start_datetime = models.DateTimeField(default=None, blank=True, null=True)
    end_datetime = models.DateTimeField(default=None, blank=True, null=True)
    channel = models.CharField(max_length=100)
    directory_path = models.CharField(max_length=255, default=None, blank=True, null=True)
    timestamp = models.DateTimeField(default=timezone.now)

    
class CountLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, null=True)
    total_count = models.PositiveIntegerField(null=True)
    hwy_count = models.PositiveIntegerField(null=True)
    msu_count = models.PositiveIntegerField(null=True)
    sm_count = models.PositiveIntegerField(null=True)
    oval_count = models.PositiveIntegerField(null=True)
    class_counts = models.JSONField(default=dict)
    class_counter = models.JSONField(default=dict)
    
    def __str__(self):
        return f"CountLog - {self.timestamp}"

class VehicleLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, null=True)
    total_count = models.PositiveIntegerField(null=True)
    hwy_count = models.PositiveIntegerField(null=True)
    msu_count = models.PositiveIntegerField(null=True)
    sm_count = models.PositiveIntegerField(null=True)
    oval_count = models.PositiveIntegerField(null=True)
    class_counts = models.JSONField(default=dict)
    class_counter = models.JSONField(default=dict)
    
    @property
    def log_date(self):
        return self.timestamp.date()
    
    def __str__(self):
        return f"VehicleLog - {self.timestamp}"

class PlateLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    filename = models.CharField(max_length=255, null=True)
    video_file = BasenameField(max_length=255, null=True)
    plate_number = models.CharField(max_length=255, null=True)
    plate_image = models.ImageField(upload_to='plate_images/', null=True)
    warped_image = models.ImageField(upload_to='warped_images/', null=True)
    frame_image = models.ImageField(upload_to='frame_images/', null=True)

    def __str__(self):
        return f"PlateLog for {self.timestamp}"
    
    def display_plate_image(self):

        if self.plate_image:
            # Open the image using Pillow
            image = Image.open(self.plate_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.plate_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
    
    def display_warped_image(self):

        if self.warped_image:
            # Open the image using Pillow
            image = Image.open(self.warped_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.warped_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_plate_image.short_description = "Plate Image"
    display_warped_image.short_description = "Warped Plate Image"
    display_frame_image.short_description = "Frame Image"

class ColorLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    filename = models.CharField(max_length=255, null=True)
    vehicle_image = models.ImageField(upload_to='vehicle_images/', null=True)
    color = models.CharField(max_length=255, null=True)
    frame_image = models.ImageField(upload_to='frame_images/', null=True)

    def __str__(self):
        return f"ColorLog for {self.timestamp}"
    
    def display_vehicle_image(self):

        if self.vehicle_image:
            # Open the image using Pillow
            image = Image.open(self.vehicle_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.vehicle_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_vehicle_image.short_description = "Vehicle Image"
    display_frame_image.short_description = "Frame Image"

class SwerveLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    filename = models.CharField(max_length=255, null=True)
    plate_number = models.CharField(max_length=255, null=True)
    video_file = BasenameField(max_length=255, null=True)
    plate_image = models.ImageField(upload_to='swerveplate_images/', null=True)
    warped_image = models.ImageField(upload_to='swervewarped_images/', null=True)
    violation = models.CharField(max_length=255, null=True)
    frame_image = models.ImageField(upload_to='swerveframe_images/', null=True)

    def __str__(self):
        return f"SwerveLog for {self.timestamp}"
    
    def display_vehicle_image(self):

        if self.vehicle_image:
            # Open the image using Pillow
            image = Image.open(self.vehicle_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.vehicle_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_vehicle_image.short_description = "Vehicle Image"
    display_frame_image.short_description = "Frame Image"

class BlockLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    filename = models.CharField(max_length=255, null=True)
    plate_number = models.CharField(max_length=255, null=True)
    video_file = BasenameField(max_length=255, null=True)
    plate_image = models.ImageField(upload_to='blockplate_images/', null=True)
    warped_image = models.ImageField(upload_to='sblockwarped_images/', null=True)
    violation = models.CharField(max_length=255, null=True)
    frame_image = models.ImageField(upload_to='blockframe_images/', null=True)

    def __str__(self):
        return f"BlockLog for {self.timestamp}"
    
    def display_vehicle_image(self):

        if self.vehicle_image:
            # Open the image using Pillow
            image = Image.open(self.vehicle_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            
            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.vehicle_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)
        
    def display_frame_image(self):

        if self.frame_image:
            # Open the image using Pillow
            image = Image.open(self.frame_image.path)
            
            # Resize the image while maintaining the aspect ratio
            max_width = 200  # Set the maximum width for display
            width_percent = (max_width / float(image.size[0]))
            new_height = int(float(image.size[1]) * float(width_percent))
            #resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)

            # Use format_html to ensure HTML is not escaped
            return format_html('<img src="{}" width="{}" height="{}" />', self.frame_image.url, max_width, new_height)
        else:
            return format_html('<img src="{}" width="{}" height="{}" />', '/static/placeholder.png', 200, 200)

    display_vehicle_image.short_description = "Vehicle Image"
    display_frame_image.short_description = "Frame Image"