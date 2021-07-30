import cv2
import json
import numpy as np

def apply_transform(image, transform):
    """
    {
        "transformation": "Crop",
        "width": 100,
        "height": 100,
        "xPosition": 0,
        "yPosition": 0,
        "gravity": "NorthWest"
    },
    {
        "transformation": "Rotate",
        "degrees": 45.3
    },
    {
        "transformation": "Resize",
        "width": 100,
        "height": 100,
        "aspect": "fit",        
    }
    """
    operations = json.loads(transform)
    for o in operations:
        if o['transformation'] == 'Rotate':
            image = rotate_image(image, int(o['Degrees']))
        if o['transformation'] == 'Crop':
            image = image[o['yPosition']:o['yPosition']+o['height'], o['xPosition']:+o['xPosition']+o['width']]
        if o['transformation'] == 'Resize':
            image = image_resize(image, width=o['width'], height=o['height'])
    return image

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a