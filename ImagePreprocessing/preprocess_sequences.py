"""Convert each seq_XX folder of 8 PNG frames into one .pt tensor
[C, D, H, W] = [3, 8, 400, 400].  Run once before training"""
import os, argparse, torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def preprocess(dataset_root, output_root, seq_len=8, img_size=(400,400)):
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    for split in ["train","val"]:
        for cls in ["Normal","DDoS"]:
            src = os.path.join(dataset_root, split, cls)
            dst = os.path.join(output_root, split, cls)
            if not os.path.exists(src): continue
            os.makedirs(dst, exist_ok=True)
            seq_dirs = [d for d in os.listdir(src) if d.startswith("seq_")]
            for seq in tqdm(seq_dirs, desc=f"{split}/{cls}"):
                frames = sorted([f for f in os.listdir(os.path.join(src,seq))
                                 if f.endswith(".png")],
                                key=lambda x:int(x.split('_')[-1].split('.')[0]))
                if len(frames)!=seq_len: continue
                imgs=[]
                for f in frames:
                    with Image.open(os.path.join(src,seq,f)) as im:
                        imgs.append(tfm(im.convert("RGB")))
                seq_tensor=torch.stack(imgs,0).permute(1,0,2,3)  # [C,D,H,W]
                torch.save(seq_tensor, os.path.join(dst,seq+".pt"))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_root",required=True)
    ap.add_argument("--output_root",required=True)
    args=ap.parse_args()
    preprocess(args.dataset_root,args.output_root)
