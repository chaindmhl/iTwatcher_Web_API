import cv2
import numpy as np
from tracking.deepsort_tric.warp_all import warp_allplate

def warp_plate_image(image):
    def find_largest_rectangle(image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and help with contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection to find edges in the image
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area, in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the largest rectangle
        largest_rectangle = None
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the polygon has four corners, it is a rectangle
            if len(approx) == 4:
                largest_rectangle = approx
                break

        return largest_rectangle

    def perspective_transform(image, corner_points):
        # Order the corner points in a clockwise order
        ordered_points = np.zeros((4, 2), dtype=np.float32)
        sum_coordinates = corner_points.sum(axis=1)
        ordered_points[0] = corner_points[np.argmin(sum_coordinates)]
        ordered_points[2] = corner_points[np.argmax(sum_coordinates)]
        diff_coordinates = np.diff(corner_points, axis=1)
        ordered_points[1] = corner_points[np.argmin(diff_coordinates)]
        ordered_points[3] = corner_points[np.argmax(diff_coordinates)]

        # Define the target width and height for the perspective transformation
        target_width = 300
        target_height = 400

        # Define the target rectangle points
        target_points = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
                                dtype=np.float32)

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered_points, target_points)

        # Apply the perspective transformation
        warped = cv2.warpPerspective(image, M, (target_width, target_height))

        return warped

    # Find the largest rectangle in the image
    largest_rectangle = find_largest_rectangle(image)

    # If a rectangle is found, perform perspective transformation
    if largest_rectangle is not None:
        # Get the corner points of the rectangle
        corner_points = largest_rectangle.reshape(4, 2)

        # Perform perspective transformation
        warped_image = perspective_transform(image, corner_points)

        return warped_image
    else:
        # print("Cannot detect edges. Trying alternative approach...")

        # Your alternative approach goes here
        alternative_warped_image = warp_allplate(image)

        if alternative_warped_image is not None:
            # print("Alternative approach successful.")
            return alternative_warped_image
        else:
            # print("Alternative approach failed. Using original image.")
            return image
