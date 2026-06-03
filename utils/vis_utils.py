import cv2
import numpy as np
import torchvision.transforms.functional as TF


def overlay_text(image, text, pos=(10, 50)):
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    cv2.putText(image, text, pos, font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image


def tensor_to_cv2(image_tensor):
    image_tensor = image_tensor.clamp(0, 1)
    image_np = TF.to_pil_image(image_tensor.cpu())
    return np.array(image_np)[:, :, ::-1]
