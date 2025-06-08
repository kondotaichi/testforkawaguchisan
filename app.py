import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np

# ハイパーパラメータ
z_dim = 100
n_classes = 10
model_path = "generator.pth"

# ラベル埋め込みクラス
class LabelEmbedding(nn.Module):
    def __init__(self, n_classes, embed_size):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_size)

    def forward(self, labels):
        return self.embed(labels)

# ジェネレータクラス
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
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = torch.cat([z, self.label_emb(labels)], dim=1)
        img = self.net(z)
        return img.view(-1, 1, 28, 28)

# 学習済みモデルの読み込みをキャッシュ
@st.cache_resource
def load_generator(path=model_path):
    G = Generator(z_dim, n_classes)
    G.load_state_dict(torch.load(path, map_location="cpu"))
    G.eval()
    return G

# Streamlit アプリケーション本体
def main():
    st.title("MNIST 手書き数字ジェネレータ")
    st.sidebar.header("設定")
    digit = st.sidebar.selectbox("生成する数字", list(range(n_classes)))
    n_imgs = st.sidebar.slider("枚数", 1, 10, 5)

    if st.sidebar.button("生成"):
        G = load_generator()
        # ノイズとラベル
        z = torch.randn(n_imgs, z_dim)
        labels = torch.full((n_imgs,), digit, dtype=torch.long)

        with torch.no_grad():
            fake_imgs = G(z, labels)

        # [–1,1] → [0,1]
        imgs = (fake_imgs + 1) / 2
        # グリッド作成
        grid = make_grid(imgs, nrow=n_imgs)
        # NumPy 配列変換
        npimg = grid.mul(255).add(0.5).clamp(0,255).byte().permute(1,2,0).numpy()
        # 幅を画像枚数に応じて設定
        display_width = 56 * n_imgs
        st.image(npimg, width=display_width, caption=f"Generated: {digit}")

if __name__ == "__main__":
    main()
