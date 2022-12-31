import copy

import cv2
from models.pspnet import PSPNet
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from utils.data_utils import fits2matrix

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0)]

img_data2 = fits2matrix('./LMC/lmc_askap_aconf.fits')
img_data2 = img_data2[0][0]
img_data2[np.isnan(img_data2)] = -1
pspnet = PSPNet(4, 8)
df2 = pd.read_csv('./csv/snrs.csv')

pspnet.load_state_dict(torch.load('ep041-loss0.247-val_loss0.264.pth', map_location=torch.device('cpu')))
pspnet.eval()
img = copy.deepcopy(img_data2).astype(np.float32)
img = torch.from_numpy(img)
img = torch.stack([img, img, img], 2)
img = np.array(img)
count = 0
for arr in df2.values[:, 2:]:
    count += 1
    x, y = int(arr[0]), int(arr[1])
    radius = int(arr[2])
    # cv2.circle(img_data2, (x, y), radius, color=(0, 0, 255), thickness=1)
    cur = img_data2[y - 90: y + 90, x - 90:x + 90].astype(np.float32)
    img = cv2.circle(img, (x, y), radius, (0, 0, 255), thickness=1)
    b = img[y - 90: y + 90, x - 90:x + 90] * 255

    a = torch.from_numpy(cur)
    print(a.size())
    # a = torch.unsqueeze(a, 0)
    a = torch.stack([a, a, a], 0)

    a = torch.unsqueeze(a, 0)
    output = pspnet(a)[1][0]
    output = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().detach().numpy()
    output = output.argmax(axis=-1)

    seg_img = np.zeros((np.shape(output)[0], np.shape(output)[1], 3))
    for c in range(4):
        seg_img[:, :, 0] += ((output[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((output[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((output[:, :] == c) * (colors[c][2])).astype('uint8')
    print(b.shape)

    b = cv2.addWeighted(b.astype(np.float64), 0.5, seg_img, 0.5, 0)
    # cv2.imwrite('imgs/' + str(count) + '_test.jpg', img_data2, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imshow('1', b)
    cv2.waitKey(0)
