import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the Generator architecture
class Generator(nn.Module):
    def __init__(self, z_dim, img_shape, n_classes):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + n_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_shape[0], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Embed labels and concatenate with the noise vector
        labels = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        z = z.unsqueeze(2).unsqueeze(3)
        input_gen = torch.cat([z, labels], dim=1)
        output = self.model(input_gen)
        output = output.view(-1, *self.img_shape)
        return output

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
img_shape = (3, 64, 64)
n_classes = 2
generator = Generator(z_dim, img_shape, n_classes).to(device)
generator.load_state_dict(torch.load('generator_epoch_90.pth', map_location=device))
generator.eval()

# Set up Streamlit interface
st.title('GAN Image Generator')
label_input = st.slider('Select a class label', 0, n_classes - 1, 0)
noise_input = st.text_input('Enter a seed for random noise (optional):', '')

if st.button('Generate'):
    if noise_input:
        torch.manual_seed(int(noise_input))
    noise = torch.randn(1, z_dim, device=device)
    labels = torch.LongTensor([label_input]).to(device)

    # Generate image
    with torch.no_grad():
        generated_image = generator(noise, labels)

    # Convert tensor to PIL Image and display
    generated_image = transforms.ToPILImage()(generated_image.squeeze(0).cpu())
    st.image(generated_image, caption='Generated Image', use_column_width=True)