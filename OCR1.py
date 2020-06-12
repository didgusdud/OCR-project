import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

print(pytesseract.image_to_string('scannedImage.png'))

print(pytesseract.image_to_string('scannedImage.png', lang='kor+eng', config='--psm 1 -c preserve_interword_spaces=1'))

