from fastapi import FastAPI, File, UploadFile
from http import HTTPStatus
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import cv2
from fastapi.responses import FileResponse

app = FastAPI()

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
def predict_step(image_paths):
   images = []
   for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
         i_image = i_image.convert(mode="RGB")

      images.append(i_image)
   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   return preds

@app.post("/generate_caption/")
async def generate_caption(data: UploadFile = File(...)):
    # Open a new file 'image.jpg' in write binary mode
    with open("image.jpg", 'wb') as image:
        # Read the contents of the uploaded file and write them to the 'image.jpg' file
        content = await data.read()
        image.write(content)

    # Load the image and extract its pixel values
    image_pil = Image.open("image.jpg")
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert(mode="RGB")
    pixel_values = feature_extractor(images=[image_pil], return_tensors="pt").pixel_values.to(device)

    # Generate a caption for the image
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Create a response dictionary with the input 'data' and the generated caption
    response = {
        "input": data.filename,
        "message": caption,
        "status-code": HTTPStatus.OK,
    }

    # Return the response dictionary as the HTTP response
    return response

