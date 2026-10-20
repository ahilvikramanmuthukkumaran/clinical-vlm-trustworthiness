from PIL import Image
from validation import validate_image, validate_question


def test_valid_question():
    assert validate_question("What is visible?")[0] is True

def test_empty_question():
    assert validate_question("")[0] is False

def test_valid_image():
    image = Image.new("RGB", (100,100))
    image.format = "PNG"
    assert validate_image(image)[0] is True

def test_valid_image_format():
    image = Image.new("RGB",(100,100))
    image.format = "BMP"

    valid, _ = validate_image(image)
    assert valid is False