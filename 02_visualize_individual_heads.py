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

intermediate = {}
def debug_hook(module, input, output):
    if isinstance(output, PIL.Image.Image):
        return output
    name = str(type(module))
    idx = len(intermediate)
    if idx == 3:
        import pdb; pdb.set_trace()
        print(input[0].sum())
    if isinstance(output, tuple):
        to_log = tuple([each.cpu().data.numpy() for each in output])
    else:
        to_log = output.cpu().data.numpy()
    log = {
        'name': name,
        'output': to_log
    }
    intermediate[idx] = log
    return output

# torch.nn.modules.module.register_module_forward_hook(debug_hook)

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
            layer_images = []
            for idx in range(each_attention_layer.size(0)):
                vis = each_attention_layer[idx, 0, 1:].reshape(7,7).detach().numpy()
                vis -= vis.min()
                vis /= vis.max()
                vis = cv2.resize(vis, (224, 224))[...,np.newaxis]
                result = (vis * image_vis).astype(np.uint8)
                layer_images.append(result)

            output_file = Path(f'logs/{image_name}/layer_{layer_n:02d}.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            show_images(layer_images, max_columns=4, show=False, output=str(output_file))

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(query)
    print("Label probs:", probs)

