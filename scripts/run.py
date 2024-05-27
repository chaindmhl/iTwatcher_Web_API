from object_tracker_cam_color import VehiclesCounting

video = './data/videos/traf.mp4' # Video's path
log_filename = 'Camera' # log_filename
line_position = 0.55 # location of line caculated by height of image
line_angle = 180

VC = VehiclesCounting(log_filename, video=video, info=True,
                      dont_show=True,
                      output='./outputs/rez2.mp4',
                      detection_line=(0.55, 180))

VC.run()

'''from color_recognition import recognize_color, process_folder_images

folder_path = './cropped_images'
process_folder_images(folder_path)'''

