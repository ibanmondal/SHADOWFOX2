# src/utils.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid




def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
os.makedirs(checkpoint_dir, exist_ok=True)
path = os.path.join(checkpoint_dir, filename)
torch.save(state, path)
if is_best:
best_path = os.path.join(checkpoint_dir, 'best_model.pth')
torch.save(state, best_path)




def imshow_batch(images, labels, classes, out_path):
# images: tensor batch; labels: tensor
images = images.cpu()
grid = make_grid(images, nrow=8, normalize=True)
npimg = grid.numpy()
plt.figure(figsize=(10,4))
plt.axis('off')
plt.imshow(np.transpose(npimg, (1,2,0)))
plt.title('Batch example: ' + ', '.join([classes[i] for i in labels[:8]]))
plt.savefig(out_path)
plt.close()