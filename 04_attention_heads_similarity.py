from pathlib import Path
from clip.clip import load
from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer
import PIL
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import types
from dawnet.data.image import show_images


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


if __name__ == '__main__':

    MODEL_PATH = 'clip.pth'
    VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'

    # model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
    # model.load_state_dict(state_dict = torch.load(MODEL_PATH))
    model, transform = load('ViT-B/32', jit=False)

    tokenizer = SimpleTokenizer(
            bpe_path=VOCAB_PATH,
            context_length=model.context_length)
    # transform = build_transform(model.input_resolution.item())
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
    # model.register_module_forward_hook(debug_hook)
    with torch.no_grad():
        query = ["a horse", "a human", "an apple", "a tiger", "a cat", "a human and a tiger"]
        # query = ["a photo of a tinca tinca", "a photo of a wombat", "a photo of a restaurant"]
        text = tokenizer.encode(query).to(device)
        text_features = model.encode_text(text)  # N_queries x 512

        # image_path = "/home/john/datasets/imagenet/object_localization/val/n01440764/ILSVRC2012_val_00002138.JPEG"
        image_path = 'images/tiger1.jpg'
        image_name = Path(image_path).stem
        image_vis = np.asarray(view_transform(Image.open(image_path)))
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image) # 1 x 512

        visual_attention = get_attention_maps(model, visual=True) #[<n_heads, t, t>]

        for layer_n, each_attention_layer in enumerate(visual_attention):
            heads = []
            for idx in range(each_attention_layer.size(0)):
                head = each_attention_layer[idx, 0, 1:].detach().numpy()
                head -= head.min()
                head /= head.max()
                head = head / np.linalg.norm(head, ord=2)
                heads.append(head)
            heads = np.stack(heads)
            result = np.matmul(heads, heads.T)
            diag = np.tril(result)
            diag[np.diag_indices_from(diag)] = 0.0
            print(f"{diag.sum() / (diag.size / 2 - diag.shape[0] / 2):.4f}")

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(query)
    print("Label probs:", probs)

