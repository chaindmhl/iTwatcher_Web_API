import cv2
import numpy as np

def warp_allplate(image):
    def find_largest_rectangle(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        largest_rectangle = None

        for contour in contours:
            if len(contour) >= 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_rectangle = contour
        
        return largest_rectangle
        
    def warp_to_straight(image, rectangle):
        # Get the dimensions of the image
        img_height, img_width = image.shape[:2]

        rect = cv2.minAreaRect(rectangle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        # Check if the rectangle dimensions meet the criteria
        if height >= 100 and width >= 200:
            # Determine the orientation of the original rectangle
            if width > height:
                dst_width = width
                dst_height = height
            else:
                dst_width = height
                dst_height = width

            src_pts = box.astype("float32")
        
            dst_pts = np.array([[0, 0], [dst_width - 1, 0], [dst_width - 1, dst_height - 1], [0, dst_height - 1]], dtype="float32")
        
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (dst_width, dst_height))

            return warped
        else:
            return None

    # # Load the image
    # image_path = '/home/itwatcher/Desktop/codes/NE.jpg'
    # original_image = cv2.imread(image_path)

    # if original_image is None:
    #     print(f"Error: Image not found at path: {image_path}")
    # else:
    #     # Find the largest rectangle
    #     largest_rectangle = find_largest_rectangle(original_image)

    #     if largest_rectangle is not None:
    #         # Warp the rectangle to make it straight if it meets the criteria
    #         straightened_image = warp_to_straight(original_image, largest_rectangle)

    #         if straightened_image is not None:
    #             # Display the original and straightened images
    #             cv2.imshow('Original Image', original_image)
    #             cv2.imshow('Straightened Image', straightened_image)
            
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #         else:
    #             print("Rectangle dimensions do not meet the criteria for straightening.")
    #     else:
    #         print("No suitable rectangle found in the image.")
