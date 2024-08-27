#Om Gole, Gabor, Period 6
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Initialize the models
D = Discriminator()
G = Generator()

# Loss and Optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training Loop
# CHANGE EPOCH RANGE TO VISUALIZE TIME AND d/g loss
for epoch in range(5):
    for real_images, _ in dataloader:
        # Train Discriminator
        D.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(real_images.size(0), 1)
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        z = torch.randn(real_images.size(0), 100)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        G.zero_grad()
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch+1}/50], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    # Save Sample Images
    if (epoch+1) % 10 == 0:
        save_image(fake_images, f'fake_images-{epoch+1}.png')

print('GAN training complete.')