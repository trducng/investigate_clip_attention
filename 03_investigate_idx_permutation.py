# please run the 03_permutate_attentions.py first
import pickle
import numpy as np

with open('logs/03_permute_attentions_idx.pkl', 'rb') as f_in:
    data = pickle.load(f_in)  # dict of each layer, tuple = (prob, layers)

def get_pred(layer_idx):
    preds = [each[0].squeeze() for each in data[layer_idx]]
    preds = np.stack(preds)
    return preds

def get_vis(layer_idx=0):
    """Get the visualization of a specific layer at specific layer
    """
    attn_heads = [
        data[layer_idx][each_item][1][layer_idx][:,0,1:]
        for each_item in range(len(data[layer_idx]))
    ]
    return attn_heads

def get_agreement(x1, x2):
    x1 = x1 / np.linalg.norm(x1, ord=2, axis=1, keepdims=True)
    x2 = x2 / np.linalg.norm(x2, ord=2, axis=1, keepdims=True)
    result = np.diag(np.matmul(x1, x2.T))
    return result

print("Permuted Layer - Accuracy")
for layer_idx in range(12):
    pred = get_pred(layer_idx)
    print(layer_idx, pred[:,0].mean())

print('Normal mean - Normal variance')
vis_normal = data['normal'][0][1]
for each_vis in vis_normal:
    print(each_vis.mean(), each_vis.var())
