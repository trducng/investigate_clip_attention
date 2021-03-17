from clip.clip import load
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer


def get_imagenet():
    # configs
    MODEL_PATH = 'clip.pth'
    VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'
    IMAGENET_PATH = '/home/john/john/data/imagenet'
    is_fp16 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize the model
    # model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
    # model.load_state_dict(state_dict = torch.load(MODEL_PATH))
    model, transform = load('ViT-B/32', jit=False)
    if is_fp16:
        model.to(device=device).eval().half()
    else:
        model.to(device=device).eval().float()

    # initializer the tokenizer + image transform
    tokenizer = SimpleTokenizer(
            bpe_path=VOCAB_PATH,
            context_length=model.context_length)
    # transform = build_transform(model.input_resolution.item())

    # initialize the data
    data = datasets.ImageNet(IMAGENET_PATH, 'val', transform=transform)
    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=16)
    # important no shuffle

    # inference
    predictions = []
    ground_truths = []
    model.eval()
    with torch.no_grad():
        query = [f'a {", ".join(each)}' for each in data.classes]
        text = tokenizer.encode(query).to(device)

        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            image_pred, text_pred = model(x, text)
            predictions += image_pred.argmax(dim=-1).cpu().data.numpy().tolist()
            # print(predictions)
            ground_truths += y.data.numpy().tolist()

            # print(idx)
            if idx % 100 == 1:
                print(idx)

    return predictions, ground_truths



if __name__ == '__main__':
    predictions, ground_truths = get_imagenet()
    with open('output_get_data.pkl', 'wb') as f_out:
        pickle.dump({'predictions': predictions, 'ground_truths': ground_truths}, f_out)
