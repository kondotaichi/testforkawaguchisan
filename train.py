import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ハイパーパラメータ
batch_size = 128
z_dim = 100
n_classes = 10
lr = 2e-4
n_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)

# データセット
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=4)

# ラベル埋め込み層
class LabelEmbedding(nn.Module):
    def __init__(self, n_classes, embed_size):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_size)
    def forward(self, labels):
        return self.embed(labels)

# ジェネレータ
class Generator(nn.Module):
    def __init__(self, z_dim, n_classes):
        super().__init__()
        self.label_emb = LabelEmbedding(n_classes, z_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    def forward(self, z, labels):
        z = torch.cat([z, self.label_emb(labels)], dim=1)
        img = self.net(z)
        return img.view(-1, 1, 28, 28)

# ディスクリミネータ
class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.label_emb = LabelEmbedding(n_classes, 28*28)
        self.net = nn.Sequential(
            nn.Linear(28*28 * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        d_in = torch.cat([img_flat, self.label_emb(labels)], dim=1)
        return self.net(d_in)

# モデル初期化
G = Generator(z_dim, n_classes).to(device)
D = Discriminator(n_classes).to(device)
criterion = nn.BCELoss()
optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ラベルテンソル
real_label = 1.
fake_label = 0.

for epoch in range(1, n_epochs + 1):
    for imgs, labels in loader:
        batch_size_curr = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)

        # ——— Discriminator を更新 ———
        D.zero_grad()
        # 本物
        output_real = D(imgs, labels).view(-1)
        loss_D_real = criterion(output_real, torch.full((batch_size_curr,), real_label, device=device))
        # 偽生成
        z = torch.randn(batch_size_curr, z_dim, device=device)
        rand_labels = torch.randint(0, n_classes, (batch_size_curr,), device=device)
        fake_imgs = G(z, rand_labels)
        output_fake = D(fake_imgs.detach(), rand_labels).view(-1)
        loss_D_fake = criterion(output_fake, torch.full((batch_size_curr,), fake_label, device=device))
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optim_D.step()

        # ——— Generator を更新 ———
        G.zero_grad()
        output = D(fake_imgs, rand_labels).view(-1)
        loss_G = criterion(output, torch.full((batch_size_curr,), real_label, device=device))
        loss_G.backward()
        optim_G.step()

    print(f"Epoch [{epoch}/{n_epochs}]  Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")
    # チェックポイント保存
    torch.save(G.state_dict(), f"checkpoints/generator_epoch{epoch}.pth")

# 最終モデルを保存
torch.save(G.state_dict(), "generator.pth")
print("Training finished, model saved as generator.pth")
