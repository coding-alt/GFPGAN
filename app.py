import gradio as gr
import cv2
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from PIL import Image
import io

# 由于Gradio处理的是PIL图像，我们需要一个辅助函数来转换格式
def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def pil_to_cv2(pil_img):
    numpy_image = np.array(pil_img)
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

def process_image(image, upscale):
    # 转换图片格式
    input_img = pil_to_cv2(image)

    # 配置模型（这里假设使用GFPGAN v1.3模型）
    model_path = '/data/GFPGAN/gfpgan/weights/GFPGANv1.3.pth'
    restorer = GFPGANer(model_path=model_path, upscale=upscale, arch='clean', channel_multiplier=2)

    # 处理图片
    _, _, restored_img = restorer.enhance(input_img, has_aligned=False, only_center_face=False, paste_back=True)

    # 转换回PIL格式以便于Gradio显示
    restored_pil_img = cv2_to_pil(restored_img)

    return restored_pil_img

iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil"), gr.Slider(minimum=1, maximum=4, value=2, label="Upscale")],
    outputs="image",
    title="GFPGAN 图片修复",
    description="上传一张图片并选择放大倍数来修复图片"
    theme='Kasien/ali_theme_custom',
    css="footer {visibility: hidden}",
    allow_flagging="never"
)

iface.launch(server_name='0.0.0.0')
