
!pip install transformers pillow torch

  from transformers import BlipProcessor, BlipForConditionalGeneration
  from PIL import Image
  import torch
  from google.colab import files
  import matplotlib.pyplot as plt

  # Select device (GPU if available)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Using device:", device)

  # Load pre-trained BLIP model and processor
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

  # Function to generate captions
  def generate_caption(image_path):
      image = Image.open(image_path).convert('RGB')
      inputs = processor(image, return_tensors="pt").to(device)

      outputs = model.generate(
          **inputs,
          max_length=30,
          num_beams=5,
          early_stopping=True
      )
      return processor.decode(outputs[0], skip_special_tokens=True)

  # Upload an image
  uploaded = files.upload()
  for filename in uploaded.keys():
      img_path = filename
      # Show image
      img = Image.open(img_path)
      plt.imshow(img)
      plt.axis("off")
      plt.show()

      # Generate and print caption
      caption = generate_caption(img_path)
      print("Generated Caption:", caption)