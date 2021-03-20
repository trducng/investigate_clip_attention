import regex
from pathlib import Path
from clip.clip import load
from clip.attention import MultiheadAttention
from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer
import PIL
from PIL import Image
import torch
from torchvision import transforms
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import types

def get_attention_maps(model, visual=True):
    if visual:
        component = model.visual.transformer
    else:   # for text
        component = model.transformer

    attention_layers = []
    for name, module in component.named_modules():
        if hasattr(module, 'attention_masks_out'):
            attention_layers.append(module.attention_masks_out)

    return attention_layers


def get_fixed_attention_head(val):

    def hook(module, input, output):
        if hasattr(module, 'skip_hook') and module.skip_hook:
            return output

        mask_out = module.attention_masks_out
        mask_in = torch.ones_like(mask_out) * val
        module.attention_masks_in = mask_in
        module.skip_hook = True
        output = module(*input, need_weights=True)
        module.skip_hook = False

        return output
    return hook

def predict(query, image_path, random_layer, value):

    MODEL_PATH = 'clip.pth'
    VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'

    model, transform = load('ViT-B/32', jit=False)
    for name, module in model.named_modules():
        if regex.match(f'visual.transformer.resblocks.\d+.attn$', name):
            module.register_forward_hook(get_fixed_attention_head(value))

    tokenizer = SimpleTokenizer(
            bpe_path=VOCAB_PATH,
            context_length=model.context_length)
    view_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert('RGB')])
    is_fp16 = False

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    if is_fp16:
        model.to(device=device).eval().half()
    else:
        model.to(device=device).eval().float()

    model.eval()
    with torch.no_grad():
        text = tokenizer.encode(query).to(device)
        text_features = model.encode_text(text)  # N_queries x 512

        # image_path = "/home/john/datasets/imagenet/object_localization/val/n01440764/ILSVRC2012_val_00002138.JPEG"
        image_name = Path(image_path).stem
        image_vis = np.asarray(view_transform(Image.open(image_path)))
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image) # 1 x 512

        visual_attention = get_attention_maps(model, visual=True) #[<n_heads, t, t>]

        # for layer_n, each_attention_layer in enumerate(visual_attention):
        #     for idx in range(each_attention_layer.size(0)):
        #         vis = each_attention_layer[idx, 0, 1:].reshape(7,7).detach().numpy()
        #         vis -= vis.min()
        #         vis /= vis.max()
        #         vis = cv2.resize(vis, (224, 224))[...,np.newaxis]
        #         result = (vis * image_vis).astype(np.uint8)
        #         output_file = Path(f'logs/{image_name}/layer_{layer_n:02d}/head_{idx:02d}.png')
        #         output_file.parent.mkdir(parents=True, exist_ok=True)
        #         Image.fromarray(result).save(str(output_file))

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        visual_attention = get_attention_maps(model, visual=True) #[<n_heads, t, t>]

    return probs

if __name__ == '__main__':
    # input
    query = ["a dog", "a human", "a zebra", "a tiger", "a cat", "a human and a tiger"]
    image_path = 'images/dog2.jpg'
    output_dir = 'logs/07_fixed_attention_head'
    value = 0.02

    probs = predict(query, image_path, None, value)
    print(round(probs.squeeze()[0], 6))
    # for each_layer in range(12):
    #     probs = predict(query, image_path, each_layer, value)
    #     print(round(probs.squeeze()[0], 6))
