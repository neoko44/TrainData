import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from RealESRGAN import RealESRGAN  # RealESRGAN modelini kendi kodunuzun olduğu klasöre göre import edin

# Düşük çözünürlüğe dönüştürülecek test klasörü ve düşük çözünürlüklü görsellerin kaydedileceği klasör
test_folder = r'C:\Users\alperen.kocabalkan\Desktop\AutoDetect\Real-ESRGAN\BSDS300\images\test'
lr_folder = r'C:\Users\alperen.kocabalkan\Desktop\lr'


# Test klasöründeki görselleri dolaşarak düşük çözünürlüklü görselleri oluşturma ve kaydetme
def custom_resize(image, max_resolution):
    width, height = image.size

    if width > height:
        new_width = max_resolution
        new_height = int(new_width * height / width)
    else:
        new_height = max_resolution
        new_width = int(new_height * width / height)

    return image.resize((new_width, new_height), Image.BICUBIC)


if not os.path.exists(lr_folder):
    os.makedirs(lr_folder)

for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    image = Image.open(image_path).convert('RGB')

    # Aspect ratio'sunu koruyarak boyutlandırma
    max_resolution = 64  # LR_IMAGE_WIDTH ve LR_IMAGE_HEIGHT yerine burada max_resolution belirliyoruz
    lr_image = custom_resize(image, max_resolution)

    lr_image_path = os.path.join(lr_folder, f'lr_{image_name}')
    lr_image.save(lr_image_path)

transform = Compose([Resize((256, 256)), ToTensor()])  # HR görüntüleri 256x256 boyutuna dönüştüren ve tensörlere çeviren bir dönüşüm

# Düşük çözünürlüklü ve yüksek çözünürlüklü görselleri yüklemek için özelleştirilmiş bir veri kümesi oluşturma
class CustomDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, transform=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.transform = transform
        self.lr_images = os.listdir(lr_folder)
        self.hr_images = os.listdir(hr_folder)

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_name = self.lr_images[idx]
        hr_image_name = self.hr_images[idx]

        lr_image_path = os.path.join(self.lr_folder, lr_image_name)
        hr_image_path = os.path.join(self.hr_folder, hr_image_name)

        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = Image.open(hr_image_path).convert('RGB')

        # Dönüşümü burada gerçekleştirin
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

# Cihazı seçin (CPU veya CUDA)
device = torch.device('cpu')

# HR klasörünün yolunu güncelleyin
hr_folder = r'C:\Users\alperen.kocabalkan\Desktop\AutoDetect\Real-ESRGAN\BSDS300\images\test'

custom_dataset = CustomDataset(lr_folder, hr_folder, transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True)


# RealESRGAN modelini oluşturun
model = RealESRGAN(device, scale=2)  # Modeli cihaza gönderin

# Kayıp işlevini tanımlayın
criterion = nn.MSELoss()  # Ortalama kare hatası (Mean Squared Error) kayıp işlevi

# Optimizasyonu tanımlayın (modelin tüm parametrelerini optimize edeceğiz)
optimizer = optim.Adam(model.model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0.0

    for lr_images, hr_images in data_loader:
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # Tahminleri yapın
        sr_images = model.model(lr_images)

        # Kaybı hesaplayın
        loss = criterion(sr_images, hr_images)

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}")


# Eğitim tamamlandı, modeli kaydedebilirsiniz
torch.save(model.model.state_dict(), 'real_esrgan_model.pth')
