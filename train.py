"""Conditional GAN for paired Lab-space image colorization."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ColorDataset(Dataset):
    def __init__(self, root: Path, size: int):
        self.files = [p for p in root.rglob("*") if p.suffix.lower() in EXTS]
        self.size = size
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        rgb = Image.open(self.files[index]).convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
        lab_np = np.asarray(rgb.convert("LAB"), dtype=np.float32)
        # PIL Lab encodes L in [0,255], a/b offset by 128. Normalize each to [-1,1].
        light = torch.from_numpy(lab_np[..., 0] / 127.5 - 1.0).unsqueeze(0)
        chroma = torch.from_numpy((lab_np[..., 1:] - 128.0) / 127.5).permute(2, 0, 1).clamp(-1, 1)
        return light, chroma, lab

class Block(nn.Module):
    def __init__(self, inp, out, down=True):
        super().__init__(); conv = nn.Conv2d if down else nn.ConvTranspose2d
        self.net = nn.Sequential(conv(inp, out, 4, 2, 1, bias=False), nn.BatchNorm2d(out), nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.downs = nn.ModuleList([nn.Conv2d(1,64,4,2,1), Block(64,128), Block(128,256), Block(256,512), Block(512,512)])
        self.ups = nn.ModuleList([Block(512,512,False), Block(1024,256,False), Block(512,128,False), Block(256,64,False)])
        self.final = nn.Sequential(nn.ConvTranspose2d(128,2,4,2,1), nn.Tanh())
    def forward(self, x):
        skips=[]; h=self.downs[0](x); skips.append(h)
        for layer in self.downs[1:]: h=layer(h); skips.append(h)
        h=self.ups[0](h)
        for layer, skip in zip(self.ups[1:], reversed(skips[:-1])): h=layer(torch.cat([h,skip],1))
        return self.final(torch.cat([h,skips[0]],1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.LeakyReLU(.2,True),
            nn.Conv2d(64,128,4,2,1,bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(.2,True),
            nn.Conv2d(128,256,4,2,1,bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(.2,True),
            nn.Conv2d(256,1,4,1,1))
    def forward(self, light, ab): return self.net(torch.cat([light,ab],1))

def preview(light, fake, real, path):
    # Convert L and ab back to Lab, then RGB for display.
    rows=[]
    for l, f, r in zip(light[:4], fake[:4], real[:4]):
        def lab_rgb(chroma):
            L=((l[0].detach().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
            ab=(chroma.detach().cpu().numpy().transpose(1,2,0)*127.5+128).clip(0,255).astype(np.uint8)
            return np.asarray(Image.fromarray(np.dstack([L,ab]), mode="LAB").convert("RGB"))
        gray=np.repeat((((l[0].cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8))[...,None],3,2)
        rows.extend([gray,lab_rgb(f),lab_rgb(r)])
    fig,axes=plt.subplots(4,3,figsize=(8,10))
    for ax,im in zip(axes.flat,rows): ax.imshow(im); ax.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--image-size",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=2e-4)
    args=p.parse_args()
    if args.image_size < 32 or args.image_size % 32: raise ValueError("image-size must be >=32 and divisible by 32")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader=DataLoader(ColorDataset(args.data,args.image_size),batch_size=args.batch_size,shuffle=True,num_workers=0)
    G,D=Generator().to(device),Discriminator().to(device)
    opt_g=torch.optim.Adam(G.parameters(),lr=args.lr,betas=(.5,.999)); opt_d=torch.optim.Adam(D.parameters(),lr=args.lr,betas=(.5,.999))
    bce=nn.BCEWithLogitsLoss(); l1=nn.L1Loss(); out=Path("artifacts"); out.mkdir(exist_ok=True)
    for epoch in range(1,args.epochs+1):
        for light,real,_ in loader:
            light,real=light.to(device),real.to(device); fake=G(light)
            opt_d.zero_grad(set_to_none=True)
            real_logits=D(light,real); fake_logits=D(light,fake.detach())
            loss_d=(bce(real_logits,torch.ones_like(real_logits))+bce(fake_logits,torch.zeros_like(fake_logits)))*.5
            loss_d.backward(); opt_d.step()
            opt_g.zero_grad(set_to_none=True); fake=G(light)
            logits=D(light,fake); loss_g=bce(logits,torch.ones_like(logits))+100*l1(fake,real)
            loss_g.backward(); opt_g.step()
        print(f"epoch {epoch}/{args.epochs} | D {loss_d.item():.3f} | G {loss_g.item():.3f}")
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            preview(light,fake,real,out/f"preview-{epoch:03d}.png")
            torch.save({"generator":G.state_dict(),"epoch":epoch},out/"generator.pt")

if __name__=="__main__": main()


