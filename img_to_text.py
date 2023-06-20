from options import *

#generate text from image
def img_to_text(url):
    if VERBOSE_MODE: print("Image to text is being generated")

    img2txt = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")
    text = img2txt(url)[0]["generated_text"]

    if VERBOSE_MODE: print("Text generated successfully")
    if VERBOSE_MODE: print(f"Text is {text}")

    return text