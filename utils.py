import cv2
from PIL import Image

def read_yml():
    fs = cv2.FileStorage("tree.yml", cv2.FILE_STORAGE_READ)
    fn = fs.getNode("tree")
    return fn

def Rotate(img, method=Image.ROTATE_180):
    img = img.transpose(method)
    return img

def Scale(img, scale=0.5):
    W = img.width
    H = img.height
    img = img.resize((int(W * 1.5), int(H * 1.5)))
    return img