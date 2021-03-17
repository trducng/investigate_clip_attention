from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer 
from PIL import Image
import torch
from torchvision import transforms

model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
model.load_state_dict(state_dict = torch.load('clip.pth'))
tokenizer = SimpleTokenizer(bpe_path='bpe_simple_vocab_16e6.txt.gz', context_length=model.context_length.item())
transform = build_transform(model.input_resolution.item())
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

with torch.no_grad():
    query = ["a balo", "a human", "a tiger", "a cat", "a human and a tiger"]
    text = tokenizer.encode(query).to(device)
    text_features = model.encode_text(text)  # N_queries x 512

    image_path = "images/balloon.jpg"
    view_transform(Image.open(image_path)).show()
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image) # 1 x 512

    logits_per_image, logits_per_text = model(image, text, return_loss=False)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs) # prints: [[0.99558276 0.00217687 0.00224036]]

