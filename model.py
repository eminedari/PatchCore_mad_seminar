" trainer code for patchcore "

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import PIL
from torchvision import transforms

from patchcore.patchcore import PatchCore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.utils
import patchcore.metrics

from data_loader import TrainDataModule, get_all_test_dataloaders
from torch.utils.data import DataLoader

from enum import Enum

results_path = "results"
os.makedirs(results_path, exist_ok=True)


on_gpu = torch.cuda.is_available()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class PatchCoreModel():

    def __init__(self, seed, split_dir,target_size, batch_size, sampling_percentage, backbone_names, layers_to_extract_from, neighbour_num, pathologies, plot_pathology, test_img_path, test_mask_path):
        self.seed = seed
        self.split_dir = split_dir
        self.target_size = target_size
        self.batch_size = batch_size
        # Set the device, send empty list if no GPU is available
        self.device = patchcore.utils.set_torch_device([0] if on_gpu else [])
        self.sampling_percentage = sampling_percentage
        self.backbone_names = backbone_names
        self.layers_to_extract_from = layers_to_extract_from
        self.neighbour_num = neighbour_num
        self.pathologies = pathologies
        self.plot_pathology = plot_pathology
        self.ensemble = False
        self.test_img_path = test_img_path
        self.test_mask_path = test_mask_path
        self.results_path = results_path
           
    def get_train_dataloaders(self):
        train_data_module = TrainDataModule(
            split_dir=self.split_dir,
            target_size=self.target_size,
            batch_size=self.batch_size)

        train_dataloader = train_data_module.train_dataloader()
        val_dataloader = train_data_module.val_dataloader()

        return train_dataloader, val_dataloader

    def get_test_dataloaders(self, image_is_dict=True):
        test_dataloaders = get_all_test_dataloaders(
            split_dir=self.split_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            image_is_dict=image_is_dict) # set this flag to True to give test images anomaly labels

        return test_dataloaders

    def set_backbone_and_layers(self):
        if len(self.backbone_names) > 1:
            self.ensemble = True
            backbone_layers = [[] for _ in range(len(self.backbone_names))]
            for layer in self.layers_to_extract_from:
                        idx = int(layer.split(".")[0])
                        layer = ".".join(layer.split(".")[1:])
                        backbone_layers[idx].append(layer)
        else:
            backbone_layers = [self.layers_to_extract_from]
        
        self.backbone_layers = backbone_layers
          
    def load_patchcores(self):
        loaded_patchcores = []
        self.set_backbone_and_layers()
        # Create a PatchCore instance for each backbone 
        for i, backbone_name in enumerate(self.backbone_names):
            backbone = patchcore.backbones.load(backbone_name)
            layers = self.backbone_layers[i]
            
            patchcore_instance = patchcore.patchcore.PatchCore(self.device)
            
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers, 
                device=self.device,
                input_shape=self.image_3d_size, 
                pretrain_embed_dimension=1024, 
                target_embed_dimension=1024,
                patchsize=3, 
                featuresampler=self.sampler,
                neighbour_num=self.neighbour_num,
                nn_method=self.nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores
     
    def train(self):
        # Create a train folder to store the results
        # save_path = os.path.join(results_path, "train")
        # os.makedirs(save_path, exist_ok=True)

        # Create the dataloaders
        train_dataloader, val_dataloader = self.get_train_dataloaders()
        examples = next(iter(train_dataloader))
        self.image_3d_size = examples[0].shape

        # Fix the seeds for reproducibility
        patchcore.utils.fix_seeds(self.seed, self.device)

        # Set sampler
        self.sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(self.sampling_percentage, self.device)
        
        # Set nn method
        self.nn_method = patchcore.common.FaissNN(on_gpu = on_gpu) # faiss_num_workers as default

        # Create the model (multiple if ensemble)
        self.PatchCore_list = self.load_patchcores()
        
        # Train the model (multiple if ensemble)
        for i, PatchCore in enumerate(self.PatchCore_list):
            print("Training model {} ({}/{})".format(self.backbone_names[i], i + 1, len(self.PatchCore_list)))
            torch.cuda.empty_cache()
            PatchCore.fit(train_dataloader)
        print("training complete")     

    def test(self,):
        # Create the dataloaders
        test_dataloaders = self.get_test_dataloaders(image_is_dict=True)

        # Create a test folder to store the results
        # save_path = os.path.join(results_path, "test")
        # os.makedirs(save_path, exist_ok=True)

        # Create dictionary to store results
        result_collect = []

        # Test the model for each disease
        for pathology in range(len(self.pathologies)):
            print("Testing on disease: ", self.pathologies[pathology])
            aggregator = {"scores": [], "segmentations": []}
            # Embed the test data (multiple if ensemble)
            for i, PatchCore in enumerate(self.PatchCore_list):
                
                scores, segmentations, labels_gt, masks_gt  = PatchCore.predict(
                    test_dataloaders[self.pathologies[pathology]])

                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            # Compute the metrics combining the results from all models
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1) + 1e-6 #for batches with only 1 image
            scores = (scores - min_scores) / (max_scores - min_scores)
            self.scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1))

            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1))
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            self.segmentations = np.mean(segmentations, axis=0)
            
            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                self.segmentations, masks_gt)

            full_pixel_auroc = pixel_scores["auroc"]
            full_pixel_threshold = pixel_scores["optimal_threshold"]


            # compute pixelwise precision, recall and F1 for segmentation with the threshold
            pixel_preds = self.segmentations > full_pixel_threshold
            pixel_gt = np.array(masks_gt).reshape(self.segmentations.shape)

            #compute pixel f1 for each image
            pixel_f1 = []
            for i in range(len(pixel_preds)):
                precision = np.sum(pixel_preds[i] * pixel_gt[i]) / (np.sum(pixel_preds[i]) + 10e-10)
                recall = np.sum(pixel_preds[i] * pixel_gt[i]) / np.sum(pixel_gt[i])
                f1 = 2 * precision * recall / (precision + recall + 10e-10)

                pixel_f1.append(f1)

            # compare with ground truth masks
            full_pixel_precision = np.sum(pixel_preds * pixel_gt) / np.sum(pixel_preds)
            full_pixel_recall = np.sum(pixel_preds * pixel_gt) / np.sum(pixel_gt)
            full_pixel_f1 = 2 * full_pixel_precision * full_pixel_recall / (full_pixel_precision + full_pixel_recall)

            # Compute accuracy for anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [self.segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            if self.pathologies[pathology] == self.plot_pathology:
                self.plot_segmentation(self.segmentations,self.scores)

            result_collect.append(
                {
                "pathology": self.pathologies[pathology],
                "full_pixel_auroc": full_pixel_auroc,
                "full_pixel_precision": full_pixel_precision,
                "full_pixel_recall": full_pixel_recall,
                "full_pixel_f1": full_pixel_f1,
                "pixel_f1": pixel_f1
                }
            )
            
        return result_collect
      
    # Helper functions for plotting
    def img_transform(self, img):
        transform_img = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),]
        transform_img = transforms.Compose(transform_img)
        img = transform_img(img)
        return np.clip(img.numpy()* 255, 0, 255).astype(np.uint8)

    def msk_transform(self, mask):
        transform_mask = [transforms.ToTensor()]
        transform_mask = transforms.Compose(transform_mask)
        mask = transform_mask(mask)
        return mask.numpy()


    def plot_segmentation(self, segmentations, scores):
        patchcore.utils.plot_segmentation_images(
            savefolder=os.path.join(self.results_path,"segmentations"),
            image_paths=self.test_img_path,
            segmentations=segmentations[:7],
            anomaly_scores=scores[:7],
            mask_paths=self.test_mask_path,
            image_transform=self.img_transform,
            mask_transform=self.msk_transform
            )    