def validate_question(question):
    if not question or not question.strip():
        return False, "Question cannot be empty."
    if len(question.strip()) > 1000:
        return False, "Question is too long."
    return True, "Question is valid."

from PIL import Image

def validate_image(image):
    if image.format not in {"PNG", "JPEG"}:
        return False, "Only PNG and JPEG images are allowed."
    if image.width <= 0 or image.height <=0:
        return False, "Invalid image dimensions."
    return True, "Image is valid."
