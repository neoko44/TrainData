import torch
from PIL import Image
from RealESRGAN import RealESRGAN

device = torch.device('cpu')

model = RealESRGAN(device, scale=2)
model.load_weights('weights/RealESRGAN_x2.pth', download=True)

path_to_image = 'inputs/perspective.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
