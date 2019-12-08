import argparse
import torch
import torchvision as tv
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type=int, choices=[0, 1], default=1,
                    help="0 for single style_ref and 1 for domain training; defualt 1")
parser.add_argument("-s", "--style", type=str,default="./starry_night.jpg",
                    help="path to style reference picture; defualt starry_night.jpg")
parser.add_argument("-a", "--artist", type=int, choices=list(range(23)), default=22,
                    help="artist domain for reference; defualt 22(Van Gogh)")
parser.add_argument("-l", "--large", type=int, choices=[0, 1], default=1,
                    help="0 for use small train_set and 1 for use large train_set; defualt 1")
parser.add_argument("-c", "--checkpoint", type=str, default=None,
                    help="checkpoint folder name; default None")


args = parser.parse_args()
style_path = args.style
use_large = args.large
ckpt_path = "./cGAN_ckpts/" + args.checkpoint if args.checkpoint is not None else None
use_domain = args.domain
artist = args.artist

content_root_dir = "//datasets/ee285f-public/flickr_landscape/"
style_root_dir = "/datasets/ee285f-public/wikiart/"

if use_domain:
    from cGAN_model.domainStyleDataSet import StyleGroupDataset
    from cGAN_model.ganNet_domain import Generator, Discriminator, CGANTrainer, CGANexp
    style_root_dir = "/datasets/ee285f-public/wikiart/"
    categories = ["forest", "lake", "city"] if use_large else ["forest"]
    train_set = StyleGroupDataset(content_root_dir, style_root_dir, 
                                  content_categories=categories, artist = artist)
    style_ref = None
else:
    from cGAN_model.styleDataSet import StyleTransDataset
    from cGAN_model.ganNet import Generator, Discriminator, CGANTrainer, CGANexp
    style_root_dir = "/datasets/ee285f-public/wikiart/wikiart/"
    train_set = StyleTransDataset(content_root_dir, style_root_dir, "city", "Art_Nouveau_Modern")
    if use_large:
        for cat in ["forest", "road"]:
            train_set += StyleTransDataset(content_root_dir, style_root_dir, cat, "Art_Nouveau_Modern")
    
    transform = tv.transforms.Compose([
            tv.transforms.Resize((150, 150)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

    style_ref = Image.open(style_path).convert('RGB')
    style_ref = transform(style_ref)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)


picNum = 1000 if use_large else 50   

cycleGan_trainer = CGANTrainer(device)
cycleGAN_exp = CGANexp(cycleGan_trainer, train_set, output_dir=ckpt_path, 
            style_ref = style_ref, picNum = picNum, 
            perform_validation_during_training=True)

print("start training")
cycleGAN_exp.run(num_epochs=200)
