import cv2
import json
import numpy as np

def apply_transform(frame, transform):
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
    if transform:
        operations = json.loads(transform)
        for o in operations:
            if o['transformation'] == 'Rotate':
                frame = image_rotate(frame, int(o['degrees']))
            if o['transformation'] == 'Crop':
                frame = frame[o['yPosition']:o['yPosition']+o['height'], o['xPosition']:+o['xPosition']+o['width']]
            if o['transformation'] == 'Resize':
                frame = image_resize(frame, width=o['width'], height=o['height'])
    return frame

def image_rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rect_as_points(rect):
    x, y, w, h = rect
    p1 = [x, y]
    p2 = [x+w, y]
    p3 = [x+w, y+h]
    p4 = [x, y+h]
    return [p1, p2, p3, p4]

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

def add_frame_tags(frame, tags):
    for k in tags:
        if len(tags[k]) > 0:
            for o in tags[k]:
                frame = cv2.drawContours(frame, [np.array(o)], 0, (255,0,0,0))
    return frame

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a