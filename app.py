import streamlit as st
import torch
import torch.nn as nn
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
        out = self.net(z)
        return out.view(-1, 1, 28, 28)

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
        # ランダムノイズとラベルを生成
        z = torch.randn(n_imgs, z_dim)
        labels = torch.full((n_imgs,), digit, dtype=torch.long)

        with torch.no_grad():
            fake_imgs = G(z, labels)

        # 出力を [0,1] にスケーリング
        imgs = ((fake_imgs + 1) / 2).cpu()

        # 画像を列に分けて表示
        cols = st.columns(n_imgs)
        for idx, col in enumerate(cols):
            img = imgs[idx].squeeze().numpy()
            col.image(img, width=56, caption=f"{digit}-{idx}")

if __name__ == "__main__":
    main()