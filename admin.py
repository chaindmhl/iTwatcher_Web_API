from django.contrib import admin
from .models import Video, PlateLog, CountLog, VehicleLog, ColorLog, NVRVideo, SwerveLog, OutputVideo, SwervingVideo, BlockingVideo, LPRVideo, ColorVideo

# Register your models here.
admin.site.register(Video)
admin.site.register(CountLog)
admin.site.register(VehicleLog)
admin.site.register(ColorLog)
admin.site.register(SwerveLog)
admin.site.register(NVRVideo)
admin.site.register(OutputVideo)
admin.site.register(LPRVideo)
admin.site.register(ColorVideo)
admin.site.register(SwervingVideo)
admin.site.register(BlockingVideo)


class PlateLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'video_file','display_frame_image', 'display_plate_image', 'display_warped_image','plate_number' )  # Include 'display_plate_image' here

admin.site.register(PlateLog, PlateLogAdmin)
