import numpy as np
import torch
import torch.nn.functional as F
from saliency_bbox import saliency_bbox

def QuanSaliencyMix(batch, quan_batch, labels, beta=1.0, n_wires=4): # Sửa lại, import dataset augmented ở notebook và đẩy vào theo batch
    lam = np.random.beta(beta, beta)
    device = batch.device
    B, C, H, W = batch.shape

    rand_index = torch.randperm(B, device=device)
    labels_a = labels.repeat_interleave(n_wires)
    labels_b = labels[rand_index].repeat_interleave(n_wires)
    source_batch = batch[rand_index].detach().clone()
    source_batch = F.interpolate(source_batch, scale_factor=0.5, mode='bilinear', align_corners=False)#.squeeze(0)

    bbx1, bby1, bbx2, bby2 = saliency_bbox(source_batch[0], lam)
    # print ("source_batch after bbox and before interleave: ", source_batch.shape)
    # source_batch = source_batch.repeat_interleave(n_wires, dim=0)
    # print ("source_batch after interleave: ", source_batch.shape)
    results = []

    for idx in range(B):
        img = batch[idx].permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        img_gray = np.dot(img_uint8, [0.299, 0.587, 0.114])

        s_masks = quan_batch[idx*4:idx*4+4] #dataset_augmented[idxs[idx]*4:idxs[idx]*4+4]
        # print ("s_masks info: ", type(s_masks))
        # print (s_masks.shape)
        # maximum = s_masks.max(axis=(1, 2), keepdims=True)
        # minimum = s_masks.min(axis=(1, 2), keepdims=True)
        maximum = s_masks.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        minimum = s_masks.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        s_masks = (s_masks - minimum) / (maximum - minimum + 1e-8)
        img_resized = F.interpolate(batch[idx].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
        for mask_tensor in s_masks:
            #mask_tensor = torch.tensor(_, dtype=torch.float32, device=device)
            masked = img_resized * mask_tensor
            results.append(masked)
            
    processed_images = torch.stack(results)
    expanded_rand_index = rand_index.repeat_interleave(n_wires)
    #print ("expanded_rand_index: ", len(expanded_rand_index))
    # print ("processed_images: ", processed_images.shape)
    # print ("source_batch: ", source_batch.shape)
    processed_images[:, :, bbx1:bbx2, bby1:bby2] = source_batch[expanded_rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    return processed_images.to(dtype=torch.float32), labels_a, labels_b, lam

if __name__=="__main__":
    print ("QuanMix ran.")