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


def get_attention_maps(model, visual=True):
    if visual:
        component = model.visual.transformer
    else:   # for text
        component = model.transformer

    attention_layers = []
    for layer in component.resblocks._modules.values():
        attention_layers.append(layer.attention_weights)

    attention_layers = torch.stack(attention_layers, dim=0) # layers x bs x head x t x t
    attention_layers = torch.mean(attention_layers, dim=2)  # layers x bs x t x t
    res_attention = torch.eye(attention_layers.size(-1))
    attention_layers += res_attention
    attention_layers /= attention_layers.sum(dim=-1, keepdim=True)

    final = torch.zeros(attention_layers.size())
    final[0] = attention_layers[0]
    for idx in range(1, final.size(0)):
        final[idx] = torch.matmul(attention_layers[idx], final[idx-1])

    return final.transpose(0, 1)    # bs x layer x key x query

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

torch.nn.modules.module.register_module_forward_hook(debug_hook)

if __name__ == '__main__':

    MODEL_PATH = 'clip.pth'
    VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'

    model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
    model.load_state_dict(state_dict = torch.load(MODEL_PATH))

    tokenizer = SimpleTokenizer(
            bpe_path=VOCAB_PATH,
            context_length=model.context_length.item())
    transform = build_transform(model.input_resolution.item())
    view_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert('RGB')])
    is_fp16 = True

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cuda'
    if is_fp16:
        model.to(device=device).eval().half()
    else:
        model.to(device=device).eval().float()

    model.eval()
    # model.register_module_forward_hook(debug_hook)
    with torch.no_grad():
        # query = ["a balo", "a human", "an apple", "a tiger", "a cat", "a human and a tiger"]
        query = ["a photo of a tinca tinca", "a photo of a wombat", "a photo of a restaurant"]
        text = tokenizer.encode(query).to(device)
        text_features = model.encode_text(text)  # N_queries x 512

        image_path = "/home/john/john/data/imagenet/val/n01440764/ILSVRC2012_val_00002138.JPEG"
        image_vis = np.asarray(view_transform(Image.open(image_path)))
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image) # 1 x 512

        text_attention = get_attention_maps(model, visual=False)
        visual_attention = get_attention_maps(model, visual=True).squeeze(0)

        vis = visual_attention[-1, 0, 1:].reshape(7,7).detach().numpy()
        vis -= vis.min()
        vis /= vis.max()
        vis = cv2.resize(vis, (224, 224))[...,np.newaxis]
        result = (vis * image_vis).astype(np.uint8)
        # Image.fromarray(result).show()

        logits_per_image, logits_per_text = model(image, text, return_loss=False)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(query)
    print("Label probs:", probs)
    # with open('here2.pkl', 'wb') as f_out:
    #     pickle.dump(intermediate, f_out)

