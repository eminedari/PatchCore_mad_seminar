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
            batch_size=self.batch_size)

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

    def test(self, selftest_dataloader):
        # Create the dataloaders
        test_dataloaders = self.get_test_dataloaders()

        # Embed test data with model
        for i in range(len(self.diseases)):
            print("Testing on disease: ", self.diseases[i])
            scores, segmentations, labels_gt, masks_gt  = self.test(test_dataloaders[self.diseases[i]])
            # Get predictions
            scores, segmentations, labels_gt, masks_gt = self.model.predict(test_dataloader)
            
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

            # TO DO: Anomaly labels missing (line 134)
            # TO DO: Metrics missing 