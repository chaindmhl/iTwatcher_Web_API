import time
import cv2
from tracking.deepsort_tric.helper.light_state import get_current_light_state

# Timer settings
red_duration = 30
yellow_duration = 5
green_duration = 30

# Initialize traffic light state
start_time = time.time()
current_state = 'red'

def update_light():
    global start_time, current_state

    elapsed_time = time.time() - start_time

    if current_state == 'red' and elapsed_time >= red_duration:
        current_state = 'green'
        start_time = time.time()
    elif current_state == 'green' and elapsed_time >= green_duration:
        current_state = 'yellow'
        start_time = time.time()
    elif current_state == 'yellow' and elapsed_time >= yellow_duration:
        current_state = 'red'
        start_time = time.time()

def get_current_state():
    return current_state

def overlay_traffic_light(frame, traffic_light_position, light_state):
    # Draw a rectangle or circle representing the traffic light
    # Example using a rectangle
    tl_width = 100  # Width of the traffic light indicator
    tl_height = 300  # Height of the traffic light indicator

    # Calculate coordinates for the traffic light indicator
    tl_top_left = (traffic_light_position[0], traffic_light_position[1])
    tl_bottom_right = (tl_top_left[0] + tl_width, tl_top_left[1] + tl_height)

    # Draw the rectangle on the frame
    cv2.rectangle(frame, tl_top_left, tl_bottom_right, (0, 0, 0), -1)  # -1 fills the rectangle

    # Display the current state of the traffic light (for demonstration)
    if light_state == 'red':
        cv2.circle(frame, (tl_top_left[0] + tl_width // 2, tl_top_left[1] + tl_height // 3), 30, (0, 0, 255), -1)
    elif light_state == 'yellow':
        cv2.circle(frame, (tl_top_left[0] + tl_width // 2, tl_top_left[1] + tl_height // 2), 30, (0, 255, 255), -1)
    elif light_state == 'green':
        cv2.circle(frame, (tl_top_left[0] + tl_width // 2, tl_top_left[1] + 2 * tl_height // 3), 30, (0, 255, 0), -1)

    return frame
