" trainer code for patchcore "

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from patchcore.patchcore import PatchCore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.utils
import patchcore.metrics

from data_loader import TrainDataModule, get_all_test_dataloaders
from torch.utils.data import DataLoader


results_path = "results"
os.makedirs(results_path, exist_ok=True)


on_gpu = torch.cuda.is_available()


class PatchCoreModel():

    def __init__(self, seed, split_dir,target_size, batch_size, sampling_percentage, backbone, layers_to_extract_from, neighbour_num, diseases):
        self.seed = seed
        self.split_dir = split_dir
        self.target_size = target_size
        self.batch_size = batch_size
        # Set the device, send empty list if no GPU is available
        self.device = patchcore.utils.set_torch_device([0] if on_gpu else [])
        self.sampling_percentage = sampling_percentage
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.neighbour_num = neighbour_num
        self.diseases = diseases
        
        self._image_3d_size = (3, self.target_size[0], self.target_size[1])

    def get_train_dataloaders(self):
        train_data_module = TrainDataModule(
            split_dir=self.split_dir,
            target_size=self.target_size,
            batch_size=self.batch_size)

        train_dataloader = train_data_module.train_dataloader()
        val_dataloader = train_data_module.val_dataloader()

        return train_dataloader, val_dataloader

    def get_test_dataloaders(self):
        test_dataloaders = get_all_test_dataloaders(
            split_dir=self.split_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            image_is_dict=True) # set this flag to True to give test images anomaly labels

        return test_dataloaders

    def train(self):
        # Create a train folder to store the results
        save_path = os.path.join(results_path, "train")
        os.makedirs(save_path, exist_ok=True)

        # Create the dataloaders
        train_dataloader, val_dataloader= self.get_train_dataloaders()

        # Fix the seeds for reproducibility
        patchcore.utils.fix_seeds(self.seed, self.device)

        # Set preferences
        self.sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(self.sampling_percentage,  self.device)
        print("sampler created")
        self.backbone = patchcore.backbones.load(self.backbone)
        print("backbone loaded")
        self.nn_method = patchcore.common.FaissNN(on_gpu = on_gpu) # faiss_num_workers as default
        print("nn method created")

        # Create the model
        self.model = PatchCore(self.device)
        
        print("created instance")
        self.model.load(
            backbone=self.backbone,
            layers_to_extract_from=self.layers_to_extract_from, # check string split
            device=self.device,
            input_shape=(self._image_3d_size), # check this
            pretrain_embed_dimension=1024, # check this
            target_embed_dimension=1024, # check this
            patchsize=3, # check this
            featuresampler=self.sampler,
            neighbour_num=self.neighbour_num,
            nn_method=self.nn_method,
        )
        print("model loaded")

        # Train the model
        self.model.fit(train_dataloader)
        print("model fitted")

    def test(self,):
        # Create the dataloaders
        test_dataloaders = self.get_test_dataloaders()

        # Create a test folder to store the results
        save_path = os.path.join(results_path, "test")
        os.makedirs(save_path, exist_ok=True)

        result_collect = {}
        # Embed test data with model
        for i in range(len(self.diseases)):
            print("Testing on disease: ", self.diseases[i])

            # Get predictions for each disease
            scores, segmentations, labels_gt, masks_gt  = self.model.predict(test_dataloaders[self.diseases[i]])
            
            # Normalize scores and segmentations
            scores = np.array(scores)
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0) 
            
            ########################### check again if anomaly_labels missing == labels_gt (line 134)############################################

            # Compute evaluation metrics
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"] # labels_gt was anomaly_labels
            
            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]     

            # Save results
            result_collect[self.diseases[i]] = {
                "segmentations": segmentations,
                "instance_auroc": auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "anomaly_pixel_auroc": anomaly_pixel_auroc,

            }       

        # Store PatchCore model for later re-use.
        model.save_to_path(results_path)

        # Calculate mean scores for each disease
        all_mean_scores = {}
        for i in range(len(self.diseases)):
            disease_scores =  result_collect[self.diseases[i]]
            mean_scores = {}
            for score in disease_scores:
                mean_scores[score] = np.mean(disease_scores[score])
            all_mean_scores[self.diseases[i]] = mean_scores

        print("All mean scores: ", all_mean_scores)

