from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage

def generate_pdf(swerve_log_instance, filename):
    # Create a PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Extract information from the SwerveLog instance
    timestamp = swerve_log_instance.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    plate_number = swerve_log_instance.plate_number
    violation = swerve_log_instance.violation

    # Create content for the PDF
    content = []
    content.append(Paragraph(f"Timestamp: {timestamp}", styles["Normal"]))
    content.append(Paragraph(f"Plate Number: {plate_number}", styles["Normal"]))
    content.append(Paragraph(f"Violation: {violation}", styles["Normal"]))

    # Include images if available
    if swerve_log_instance.plate_image:
        plate_image_path = swerve_log_instance.plate_image.path
        resized_plate_image_path = resize_image(plate_image_path)
        content.append(Spacer(1, 12))
        content.append(Image(resized_plate_image_path, width=400, height=300))

    if swerve_log_instance.frame_image:
        frame_image_path = swerve_log_instance.frame_image.path
        resized_frame_image_path = resize_image(frame_image_path)
        content.append(Spacer(1, 12))
        content.append(Image(resized_frame_image_path, width=400, height=300))

    # Build the PDF document
    doc.build(content)

def resize_image(image_path, max_width=400, max_height=300):
    image = PILImage.open(image_path)
    image.thumbnail((max_width, max_height), PILImage.ANTIALIAS)
    resized_image_path = image_path.replace(".jpg", "_resized.jpg")  # Change file extension as needed
    image.save(resized_image_path)
    return resized_image_path
