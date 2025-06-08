import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# ===== ハイパーパラメータ =====
z_dim = 20
n_classes = 10
model_path = "cvae.pth"

# ===== CVAE 定義（train.py と同一） =====
class CVAE(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(n_classes, n_classes)
        # エンコーダ
        self.fc1 = nn.Linear(784 + n_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        # デコーダ
        self.fc3 = nn.Linear(latent_dim + n_classes, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, labels):
        h = torch.cat([x, self.label_emb(labels)], dim=1)
        h = F.relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        h = torch.cat([z, self.label_emb(labels)], dim=1)
        h = F.relu(self.fc3(h))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparam(mu, logvar)
        return self.decode(z, labels), mu, logvar

# ===== モデル読み込み =====
@st.cache_resource

def load_model(path=model_path):
    model = CVAE(z_dim, n_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# ===== Streamlit UI =====
st.set_page_config(page_title="MNIST CVAE ジェネレータ", layout="wide")
st.title("MNIST 手書き数字ジェネレータ (VAE)")

# サイドバー入力
digit = st.sidebar.selectbox("生成する数字", list(range(n_classes)))
n_imgs = st.sidebar.slider("枚数", 1, 10, 5)

# 生成ボタン
if st.sidebar.button("生成"):
    # モデルロード
    model = load_model()
    # ランダム潜在ベクトルとラベル
    z = torch.randn(n_imgs, z_dim)
    labels = torch.full((n_imgs,), digit, dtype=torch.long)
    # 推論
    with torch.no_grad():
        recon = model.decode(z, labels)
    # 再構築画像を [n,28,28] へ
    imgs = recon.view(-1, 28, 28).numpy()
    # カラム分割して表示
    cols = st.columns(n_imgs)
    for idx, col in enumerate(cols):
        img = (imgs[idx] * 255).astype(np.uint8)
        col.image(img, width=56, caption=f"{digit}-{idx}")
