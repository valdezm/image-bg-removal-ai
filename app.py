import gradio as gr
from loadimg import load_img
from PIL import Image
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    processed_image = process(im)
    return (processed_image, origin)

@spaces.GPU
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)

    # Crop to non-transparent bounding box
    bbox = image.getbbox()
    padding_pct = 0.10  # 10% of bbox size
    if bbox:
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        pad_w = int(bbox_width * padding_pct)
        pad_h = int(bbox_height * padding_pct)
        left = max(bbox[0] - pad_w, 0)
        upper = max(bbox[1] - pad_h, 0)
        right = min(bbox[2] + pad_w, image.width)
        lower = min(bbox[3] + pad_h, image.height)
        expanded_bbox = (left, upper, right, lower)
        cropped = image.crop(expanded_bbox)
    else:
        cropped = image

    # Place on white background
    white_bg = Image.new("RGB", cropped.size, (255, 255, 255))
    white_bg.paste(cropped, mask=cropped.split()[-1])  # use alpha as mask
    # Add transparent logo watermark
    try:
        logo = Image.open("logo-t.png").convert("RGBA")
        # Resize logo if wider than 25% of output image
        max_logo_w = int(white_bg.width * 0.25)
        if logo.width > max_logo_w:
            logo_h = int((max_logo_w / logo.width) * logo.height)
            logo = logo.resize((max_logo_w, logo_h), Image.Resampling.LANCZOS)
        # Optional: adjust transparency
        # alpha = logo.split()[3].point(lambda x: int(x * 0.7))  # 70% opacity
        # logo.putalpha(alpha)
        # Position: bottom-right with 10px margin
        x = white_bg.width - logo.width - 10
        y = white_bg.height - logo.height - 10
        white_bg.paste(logo, (x, y), logo)
    except Exception as e:
        print(f"Watermark error: {e}")
    return white_bg

def process_file(f):
    name_path = f.rsplit(".", 1)[0] + ".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

slider1 = gr.ImageSlider(label="Processed Image", type="pil", format="png")
slider2 = gr.ImageSlider(label="Processed Image from URL", type="pil", format="png")
image_upload = gr.Image(label="Upload an image")
image_file_upload = gr.Image(label="Upload an image", type="filepath")
url_input = gr.Textbox(label="Paste an image URL")
output_file = gr.File(label="Output PNG File")

# Example images
chameleon = load_img("butterfly.jpg", output_type="pil")
url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

tab1 = gr.Interface(fn, inputs=image_upload, outputs=slider1, examples=[chameleon], api_name="image")
tab2 = gr.Interface(fn, inputs=url_input, outputs=slider2, examples=[url_example], api_name="text")
tab3 = gr.Interface(process_file, inputs=image_file_upload, outputs=output_file, examples=["butterfly.jpg"], api_name="png")

demo = gr.TabbedInterface(
    [tab1, tab2, tab3], ["Image Upload", "URL Input", "File Output"], title="Background Removal Tool"
)

if __name__ == "__main__":
    demo.launch(show_error=True)