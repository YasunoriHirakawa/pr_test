import os
from glob import iglob

import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import models


class ImageFeatures:

    def __init__(self, image, path):

        self.num_devides = 4
        self.input_image_size = 224
        self.path = path
        self.all_keypoints = self.detect_keypoints(image)
        self.image_with_keypoints = cv2.drawKeypoints(np.array(image), self.all_keypoints, None)
        self.features = self.extract_feature_vector(image)

    def extract_feature_vector(self, image):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = models.resnet18(pretrained=True).to(device)

        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.fc = nn.Identity()

        images = []
        for i in range(self.num_devides):
            pixel_origin = i * self.input_image_size
            cropped_image = image.crop((pixel_origin, 0, pixel_origin + self.input_image_size, self.input_image_size))
            keypoints = self.detect_keypoints(cropped_image)
            if len(keypoints) >= len(self.all_keypoints) / 10:
                images.append(to_tensor(cropped_image).to(device))

        return feature_extractor(torch.stack(images, 0))

    def detect_keypoints(self, image):

        detector = cv2.AKAZE_create()

        converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        keypoints = detector.detect(converted_image)

        return keypoints


def calc_similarity(reference_data, query_features):

    cosine_similarity = nn.CosineSimilarity(dim=0) 
    
    similarities = []
    paths = []
    images = []
    for reference_features in reference_data:
        similarities_ = []
        for reference_feature in reference_features.features:
            for query_feature in query_features.features:
                similarities_.append(cosine_similarity(reference_feature, query_feature))
        similarities.append(max(similarities_))
        paths.append(reference_features.path)
        images.append(reference_features.image_with_keypoints)

    data = []
    for similarity, path, image in zip(similarities, paths, images):
        normalized_similarity = (similarity - min(similarities)) / (max(similarities) - min(similarities))
        data.append({"sim": normalized_similarity.item(), "ref": path, "image": image})

    sorted_data = sorted(data, key=lambda x:x["sim"], reverse=True)
    print_data(sorted_data, query_features)


def print_data(data, query):

    num_showing = 3
    fig = plt.figure(tight_layout=True)

    ax = fig.add_subplot(num_showing+1, 1, 1, title="Query")
    ax.imshow(query.image_with_keypoints)

    for i, d in enumerate(data):
        if i >= num_showing:
            break
        ax = fig.add_subplot(num_showing+1, 1, i+2, title=f"sim:{d['sim']}")
        ax.imshow(d["image"])
    
    plt.show()


def main():

    reference_data = []
    for path in iglob("reference_images/*"):
        image = Image.open(path)
        reference_data.append(ImageFeatures(image, path))

    query_features = []
    for path in iglob("query_images/*"):
        image = Image.open(path)
        query_features = ImageFeatures(image, path)

        calc_similarity(reference_data, query_features)


if __name__ == "__main__":
    main()
