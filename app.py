import streamlit as st
import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image

# モデル定義（train.py と同じクラスをコピペ）
# …（Generator クラス定義をここに貼る）…

@st.cache_resource
def load_generator(path="generator.pth"):
    G = Generator(z_dim=100, n_classes=10)
    G.load_state_dict(torch.load(path, map_location="cpu"))
    G.eval()
    return G

st.title("MNIST 手書き数字ジェネレータ")
st.sidebar.header("設定")
digit = st.sidebar.selectbox("生成する数字", list(range(10)))
n_imgs = st.sidebar.slider("枚数", min_value=1, max_value=10, value=5)

if st.sidebar.button("生成"):
    G = load_generator()
    z = torch.randn(n_imgs, 100)
    labels = torch.full((n_imgs,), digit, dtype=torch.long)
    with torch.no_grad():
        fake_imgs = G(z, labels)
    # 画素値を [0,1] にスケール変換
    imgs = (fake_imgs + 1) / 2
    grid = make_grid(imgs, nrow=n_imgs, normalize=False)
    npimg = grid.mul(255).add(0.5).clamp(0,255).byte().permute(1,2,0).numpy()
    st.image(npimg, caption=[str(digit)]*n_imgs, width=56)  # MNIST サイズの拡大表示

