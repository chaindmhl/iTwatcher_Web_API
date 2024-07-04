from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DownloadRequestListCreateView, DownloadRequestDetailView, success_page
from tracking.views import (
    VideoUploadViewSet,
    ProcessTrikeViewSet,
    CatchAllViewSet,
    LPRTrikeViewSet,
    LPRAllViewSet,
    CombiViewSet,
    LPRCombiViewSet,
    ColorViewSet,
    RedLightViewSet,
    BlockingViewSet,
    MyView,
    UploadView,
    PlateView,
    FrameView,
    FrameColorView,
    FrameSwerveView,
    FrameBlockView,
    SwerveView,
    BlockView,
    MapView,
    ColorView,
    CountLogViewSet,
    CountLogListView,
    TrikeVehicleLogListView,
    VehicleLogListView,
    TricycleCountGraphView,
    VehicleCountGraphView,
    SignupView,
    CustomLoginView,
    DarknetTrainView,
    LPRView,
    TrackCountView,
    ColorRecognitionView,
    VioDetectionView,
    UploadView,
    generate_report,
    update_plate_number
)

router = DefaultRouter()
router.register("tracking/video", VideoUploadViewSet, basename="tracking-video")
router.register("tracking/color", ColorViewSet, basename="tracking-color")
router.register("tracking/count-logs", CountLogViewSet, basename="tracking-count")
router.register("tracking/tric", ProcessTrikeViewSet, basename="tracking-tric")
router.register("tracking/catchall", CatchAllViewSet, basename="tracking-catchall")
router.register("tracking/combi", CombiViewSet, basename="tracking-combi")
router.register("tracking/lpr_trike", LPRTrikeViewSet, basename="LPR-tric")
# router.register("tracking/lpr-cam", LPRTrikeViewSet, basename="LPR-tric-cam")
router.register("tracking/lpr_all", LPRAllViewSet, basename="LPR-All_Vehicle")
router.register("tracking/lpr_combi", LPRCombiViewSet, basename="LPR-combi")
router.register("tracking/redlight", RedLightViewSet, basename="tracking-redlight")
router.register("tracking/blocking", BlockingViewSet, basename="tracking-blocking")

urlpatterns = [
    path('', include(router.urls)),
    path('my-url/', MyView.as_view(), name='my-view'),
    path('download-requests/', DownloadRequestListCreateView.as_view(), name='downloadrequest-list-create'),
    path('upload-requests/', UploadView.as_view(), name='upload-video'),
    path('download-requests/<int:pk>/', DownloadRequestDetailView.as_view(), name='downloadrequest-detail'),
    path('success/', success_page, name='success-page'),
    path('display_plates/', PlateView.as_view(), name='display_plates'),
    path('display_color/', ColorView.as_view(), name='display_color'),
    path('redlight/', SwerveView.as_view(), name='redlight_list'),
    path('blocking/', BlockView.as_view(), name='blocking_list'),
    path('view_frame/<int:log_id>/', FrameView.view_frame, name='view_frame'),
    path('view_colorframe/<int:log_id>/', FrameColorView.view_colorframe, name='view_colorframe'),
    path('view_swerveframe/<int:log_id>/', FrameSwerveView.view_swerveframe, name='view_swerveframe'),
    path('view_blockframe/<int:log_id>/', FrameBlockView.view_blockframe, name='view_blockframe'),
    path('view_camera_map/', MapView.view_camera_map, name='view_camera_map'),
    path('count_logs/', CountLogListView.as_view(), name='count_log_list'),
    path('vehicle_logs/', VehicleLogListView.as_view(), name='vehicle_log_list'),
    path('trikeall_logs/', TrikeVehicleLogListView.as_view(), name='trikeall_log_list'),
    path('tricycle_count_graph/<int:log_id>/', TricycleCountGraphView.as_view(), name='tricycle_count_graph'),
    path('vehicle_count_graph/<str:log_date>/<int:log_id>/', VehicleCountGraphView.as_view(), name='vehicle_count_graph'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('train/', DarknetTrainView.as_view(), name='train'),
    path('lpr/', LPRView.as_view(), name='lpr-view'),
    path('track/', TrackCountView.as_view(), name='track-count'),
    path('color/', ColorRecognitionView.as_view(), name='color'),
    path('violation/', VioDetectionView.as_view(), name='viodetection'),
    path('upload-video/', UploadView.as_view(), name='upload-video'),
    path('swerve_report/<int:log_id>/', generate_report, name='generate_report'),
    path('update-plate-number/', update_plate_number, name='update_plate_number'),
    path('control-traffic-light/', RedLightViewSet.as_view({'post': 'control_traffic_light'}), name='control-traffic-light'),  # Add this line
    

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
