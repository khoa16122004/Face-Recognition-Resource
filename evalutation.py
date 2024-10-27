# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
# from insightface.model_zoo import ArcFaceONNX
# from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dataset import LFW_EVALUATION, LFW
import onnxruntime as ort
# from mtcnn import MTCNN
import numpy as np
from get_architech import get_model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *
from attack import *
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pickle as pkl
# from pySR import SymbolicRegressionModule
from torchvision import models
from torch import nn

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def calculate_similarity(preds_1, preds_2):
    similarity_scores = F.cosine_similarity(preds_1, preds_2, dim=1)
    return similarity_scores


def evaluation(model, loader, transform=None, attack=None):
    test_acc = AverageMeter()
    model.eval()
    for (img_1, img_2, mask_1, mask_2, labels) in tqdm(loader):
        img_1, img_2, mask_1, mask_2, labels = img_1.cuda(), img_2.cuda(), mask_1.cuda(), mask_2.cuda(), labels.cuda()
        
        # if attack == "fill_lips":
        #     img_1 = fill_lips(img_1, mask_1)
        #     # save_image(img_1[0, :, :, :], "test.png")
        # elif attack == "color":
        #     img_1 = color_aware_perturbation(img_1, mask_1, model)        
        # if transform:
        #     img_1 = transform(img_1)
        #     img_2 = transform(img_2)
            
        preds_1 = model(img_1)
        preds_2 = model(img_2)
        sims = calculate_similarity(preds_1, preds_2)
        acc = accuracy_FR(sims, labels)
        test_acc.update(acc, img_1.shape[0])
        # break
    print("Face Verification Acc: ", test_acc.avg)
    print(test_acc.count - test_acc.sum)