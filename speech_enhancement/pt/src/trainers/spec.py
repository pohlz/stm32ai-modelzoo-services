# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

'''Trainer for Models that take magnitude spectrograms as input and output masks
   that are applied to the complex spectrogram. Models are expected to have input shape
   (batch, frame_length, sequence_length), e.g. for n_fft=512 with 20 spectrogram frames
   (batch, 257, 20)
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
import numpy as np
from speech_enhancement.pt.src.trainers import BaseTrainer
from tqdm import tqdm
from torch.utils.data import default_collate
from pesq import pesq
from pystoi import stoi
from speech_enhancement.pt.src.metrics import si_snr, snr, SISNRLoss, SNRLoss
from pathlib import Path

class OutputHook(list):
    """ Hook to capture module outputs."""
    def __call__(self, module, input, output):
        self.append(output)

class MagSpecTrainer(BaseTrainer):
    '''Trainer class for the STFT-TCNN model.
       Model input and output shape should be (batch, n_fft // 2 + 1, sequence_length).
       Input audio clips are trimmed to the length of the shortest clip in the batch, 
       so this tends to work better with small batch sizes.
       Training metrics are training loss.
       Validation metrics are PESQ, STOI, SNR, SI-SNR and MSE between clean and denoised waveforms.
    '''
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_data: DataLoader,
                 valid_data: DataLoader,
                 frame_length: int,
                 hop_length: int,
                 n_fft: int,
                 center: bool, 
                 sampling_rate: int,
                 window: str = "hann",
                 loss: str = "spec_mse",
                 batching_strat: str = "trim",
                 weight_clipping_max: float = None,
                 activation_regularization: float = None,
                 act_reg_layer_names: list[str] = None,
                 act_reg_layer_types: list = None,
                 act_reg_threshold: float = None,
                 penalty_type: str = "l2",
                 device: str = "cuda:0",
                 save_every: int = 20,
                 ckpt_path: str = "checkpoints/",
                 logs_path: str = "training_logs.csv",
                 snapshot_path: str = "snapshot.pth",
                 device_memory_fraction: float = 0.5,
                 early_stopping: bool = False,
                 reference_metric: str = "pesq",
                 early_stopping_patience: int = 20
                 ):
        super().__init__(model=model,
                         optimizer=optimizer,
                         train_data=train_data,
                         valid_data=valid_data,
                         device=device,
                         save_every=save_every,
                         ckpt_path=ckpt_path,
                         logs_path=logs_path,
                         snapshot_path=snapshot_path,
                         device_memory_fraction=device_memory_fraction)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.center = center
        self.loss = loss
        self.window = window
        self.weight_clipping_max = weight_clipping_max
        self.activation_regularization = activation_regularization
        self.penalty_type = penalty_type
        self.act_reg_layer_names = act_reg_layer_names
        self.act_reg_layer_types = act_reg_layer_types
        self.act_reg_threshold = act_reg_threshold

        allowed_losses = ["spec_mse", "wave_mse", "wave_snr", "wave_sisnr"]
        assert self.loss in allowed_losses, f"self.loss must be one of {allowed_losses}, was {self.loss}"
        
        self.batching_strat = batching_strat
        assert self.batching_strat in ["trim", "pad"], f"self.batching_strat must be one of 'trim', 'pad', was {self.batching_strat}"
        
        self.sampling_rate = sampling_rate # Needed for PESQ computation

        # Initialize metrics header, and handle reference metric / early stopping params
        self.header = ["train_loss", "val_mse", "pesq", "stoi", "snr", "si-snr"]
        self.early_stopping = early_stopping
        self.reference_metric= reference_metric
        self.early_stopping_patience = early_stopping_patience
        assert self.reference_metric in self.header, f"Reference metric unavailable, must be one of{self.header}, was {reference_metric}"
        self.best_metric = np.inf if self.reference_metric in ["train_loss", "val_mse"] else -np.inf
        self.best_epoch = 0
        self.best_model_state_dict_path = Path(self.ckpt_path, "best_model_state_dict.pth")
        if self.loss in ["spec_mse", "wave_mse"]:
            self.loss_function = nn.MSELoss(reduction="mean")

        elif self.loss == "wave_sisnr":
            self.loss_function = SISNRLoss(reduction="mean")
        elif self.loss == "wave_snr":
            self.loss_function = SNRLoss(reduction="mean")

        if type(self.window) not in [np.ndarray, torch.Tensor]:
            # If window is a string or a tuple, pass to librosa.filters.get_window
            # instead of trying to match it to one of the window functions
            # implemented in torch
            self.window = torch.Tensor(librosa.filters.get_window(self.window, Nx=self.frame_length))
        # And move window to device
        self.window = self.window.to(self.device)
        
        # Change dataloader collate function
        if self.batching_strat == "trim":
            self.train_data.collate_fn = self._trim_collate
        elif self.batching_strat == "pad":
            self.train_data.collate_fn = self._zero_pad_collate

        if self.activation_regularization:
            self._attach_regularization_hooks(layer_names=self.act_reg_layer_names,
                                              layer_types=self.act_reg_layer_types)
        if self.penalty_type == "l1":
            self.ord = 1
        elif self.penalty_type == "l2":
            self.ord = 2
        else:
            raise ValueError(f"penalty_type must be one of 'l1', 'l2', was {self.penalty_type}")
    def _run_train_epoch(self, epoch):
        print(f"========= EPOCH {epoch + 1} : training ============")
        epoch_loss = 0
        self.model.train()
        for batch in tqdm(self.train_data):
            batch_loss = self._run_train_batch(batch)
            epoch_loss += batch_loss
        # Normalize by n° of batches
        # Note that this means we're displaying mean of means which is a bit different from 
        # mean of samples.
        epoch_loss = epoch_loss / len(self.train_data)
        print(f"========= EPOCH {epoch + 1} training loss : {epoch_loss} ============")
        self.metrics_array[epoch][0] = epoch_loss

    def _run_train_batch(self, batch):
        self.optimizer.zero_grad()

        # Clear regularization hooks
        if self.activation_regularization:
            for h in self.hooks:
                h.clear()

        if self.batching_strat == "pad":
            noisy_frames, clean_signal, sequence_lengths = batch
            noisy_frames, clean_signal = noisy_frames.to(self.device), clean_signal.to(self.device)
        else:
            noisy_frames, clean_signal = batch
            noisy_frames, clean_signal = noisy_frames.to(self.device), clean_signal.to(self.device)

        # Convert noisy complex spectrogram to magnitude spectrogram
        noisy_frames_mag = torch.abs(noisy_frames)
        pred_weighted_mask = self.model(noisy_frames_mag)

        if self.batching_strat == "pad":
            # If batching with zero-padding, apply a loss mask to the model output
            # so that we don't compute the loss on pad frames.
            loss_mask = self._loss_mask(noisy_frames.shape, sequence_lengths=sequence_lengths)
            loss_mask = loss_mask.to(self.device)
            masked_pred_weighted_mask = pred_weighted_mask * loss_mask
            pred_frames = noisy_frames * masked_pred_weighted_mask

        else:
            pred_frames = noisy_frames * pred_weighted_mask

        if self.loss == "spec_mse":
            # Can't have MSE on complex values in torch
            # Compute loss on real & imaginary parts together and then sum
            loss_r = self.loss_function(pred_frames.real, clean_signal.real)
            loss_i = self.loss_function(pred_frames.imag, clean_signal.imag)
            loss = loss_r + loss_i

        elif self.loss in ["wave_mse", "wave_sisnr", "wave_snr"]:
            pred_wave = torch.istft(pred_frames, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.frame_length, window=self.window, center=self.center)
            # Here, clean_signal should actually be a wave batch of shape (batch, wave_length)
            # Clip clean signal to length of recomposed preds
            clean_signal = clean_signal[:, :pred_wave.shape[-1]]
            loss = self.loss_function(pred_wave, clean_signal)

        if self.activation_regularization:
            reg_penalty = 0
            for h in self.hooks:
                for out in h:
                    if self.act_reg_threshold:
                        reg_mask = (torch.abs(out) >= self.act_reg_threshold)
                        out = out * reg_mask
                    reg_penalty += torch.norm(out, self.ord)
            reg_penalty *= self.activation_regularization
            loss += reg_penalty
        
        loss.backward()
        self.optimizer.step()
        # Clip weights after gradient update
        if self.weight_clipping_max:
            for p in self.model.parameters():
                p.data = p.data.clamp(min=-self.weight_clipping_max, max=self.weight_clipping_max)

        return loss

    def _run_validation_epoch(self, epoch):
        # We expect the batch size in the validation dataloader to be 1, 
        # so 1 batch corresponds to 1 noisy/clean pair, and we're not doing 
        # any padding or trimming for the sake of batching like during training
        print(f"========= EPOCH {epoch + 1} : validation ============")
        self.model.eval()
        with torch.no_grad():
            num_batches = len(self.valid_data)
            # Should put all this into a dict probably
            valid_metrics = {
                "train_loss": 0,
                "pesq": 0,
                "stoi": 0,
                "snr": 0,
                "si-snr": 0
            }
            for batch in tqdm(self.valid_data):
                batch_loss, batch_pesq, batch_stoi, batch_snr, batch_si_snr = self._run_validation_batch(batch)
                valid_metrics["train_loss"] += batch_loss
                valid_metrics["pesq"] += batch_pesq
                valid_metrics["stoi"] += batch_stoi
                valid_metrics["snr"] += batch_snr
                valid_metrics["si-snr"] += batch_si_snr

            valid_metrics["train_loss"] /= num_batches
            valid_metrics["pesq"] /= num_batches
            valid_metrics["stoi"] /= num_batches
            valid_metrics["snr"] /= num_batches
            valid_metrics["si-snr"] /= num_batches
        
        print(f"Validation MSE : {valid_metrics['train_loss']} \n"
              f"Validation PESQ : {valid_metrics['pesq']} \n"
              f"Validation STOI : {valid_metrics['stoi']} \n"
              f"Validation SNR : {valid_metrics['snr']} \n"
              f"Valid SI-SNR : {valid_metrics['si-snr']} \n")

        self.metrics_array[epoch][1:] = [valid_metrics["train_loss"], valid_metrics["pesq"], valid_metrics["stoi"], valid_metrics["snr"], valid_metrics["si-snr"]]
        
        # Early stopping stuff
        # Update the best model, best ref metric and best epoch
        self._update_best_model(value=valid_metrics[self.reference_metric], epoch=epoch)
        # If early stopping is enabled and patience is exceeded, return that we need to stop training
        return (self.early_stopping and (epoch - self.best_epoch > self.early_stopping_patience))


    def _run_validation_batch(self, batch):
        noisy_frames, clean_wave = batch
        noisy_frames = noisy_frames.to(self.device)

        # Convert noisy complex spectrogram to magnitude spectrogram
        noisy_frames_mag = torch.abs(noisy_frames)
        pred_weighted_mask = self.model(noisy_frames_mag)
        pred_frames = noisy_frames * pred_weighted_mask
        pred_wave = torch.istft(pred_frames, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.frame_length, window=self.window, center=self.center)
        # Squeeze waves
        pred_wave, clean_wave = pred_wave.squeeze(), clean_wave.squeeze()
        # Trim clean wave to the length of predicted wave
        clean_wave = clean_wave[:pred_wave.shape[-1]]
        # Put everything back on CPU
        # Then compute metrics
        denoised, clean_source = pred_wave.to("cpu"), clean_wave.to("cpu")
        # Back to numpy
        denoised = denoised.numpy()
        clean_source = clean_source.numpy()

        valid_loss = np.mean((denoised - clean_source) ** 2)

        valid_pesq = pesq(fs=self.sampling_rate,
                          ref=clean_source,
                          deg=denoised,
                          mode="wb")
        valid_stoi = stoi(x=clean_source,
                          y=denoised,
                          fs_sig=self.sampling_rate)
        valid_snr = snr(ref=clean_source, deg=denoised)
        valid_si_snr = si_snr(ref=clean_source,
                              deg=denoised)
        return (valid_loss, valid_pesq, valid_stoi, valid_snr, valid_si_snr)


    @staticmethod
    def _trim_collate(batch):
        # Intended as a replacement for the default collate_fn of a dataloader
        # Trims the end of sequences, so that the length of all sequences match the shortest in the batch.
        # Assumes batch is a (noisy_frames, clean_frames) tuple of numpy arrays
        # Assumes sequence length axis is the last axis in both arrays
        min_noisy_seq_len = min([elem[0].shape[-1] for elem in batch])
        min_clean_seq_len = min([elem[1].shape[-1] for elem in batch])
        for i, elem in enumerate(batch):
            # Convert to tensor if needed
            if isinstance(elem[0], np.ndarray):
                noisy = torch.as_tensor(elem[0])
            else:
                noisy = elem[0]
            if isinstance(elem[1], np.ndarray):
                clean = torch.as_tensor(elem[1])
            else:
                clean = elem[1]

            # Trim both noisy and clean sequences
            trimmed_noisy = torch.narrow(noisy, dim=-1, start=0, length=min_noisy_seq_len)
            trimmed_clean = torch.narrow(clean, dim=-1, start=0, length=min_clean_seq_len)

            batch[i] = (trimmed_noisy, trimmed_clean)
        # Give trimmed batch back to default dataloader collate function
        return default_collate(batch)
    
    @staticmethod
    def _zero_pad_collate(batch):
    # Intended as a replacement for the default collate_fn of a dataloader
    # Zero-pads sequences to the right, with the max length being the longest length of the batch.
    # Assumes batch is a (noisy_frames, clean_frames) tuple of numpy arrays
        max_noisy_seq_len = max([elem[0].shape[-1] for elem in batch])
        max_clean_seq_len = max([elem[1].shape[-1] for elem in batch])

        for i, elem in enumerate(batch):
            # Convert to tensor if needed
            if isinstance(elem[0], np.ndarray):
                noisy = torch.as_tensor(elem[0])
            else:
                noisy = elem[0]
            if isinstance(elem[1], np.ndarray):
                clean = torch.as_tensor(elem[1])
            else:
                clean = elem[1]

            # Pad both noisy and clean sequences
            noisy_seq_len = noisy.shape[-1]
            clean_seq_len = clean.shape[-1]

            noisy_pad_lengths = (0, max_noisy_seq_len - noisy_seq_len)
            clean_pad_lengths = (0, max_clean_seq_len - clean_seq_len)
            # Assuming noisy & clean sequences have same length
            # nn.functional.pad pads starting from the last axis
            padded_noisy = nn.functional.pad(noisy, noisy_pad_lengths, mode="constant")
            padded_clean = nn.functional.pad(clean, clean_pad_lengths, mode="constant")
            # Add original noisy sequence lengths in batch. We need this info to compute
            # the loss mask later.
            batch[i] = (padded_noisy, padded_clean, noisy_seq_len)
        # Give padded batch back to default dataloader collate function
        return default_collate(batch)
    
    @staticmethod
    # Loss mask : we don't want to compute loss on pad frames.
    def _loss_mask(batch_shape, sequence_lengths):
        mask = torch.zeros(batch_shape, requires_grad=False)
        # Manually setting loss mask to 1 on frames that are not pad frames
        for k, seq_len in enumerate(sequence_lengths):
            mask[k, :, :seq_len] += 1.0
        return mask

    def _attach_regularization_hooks(self, layer_names=None, layer_types=None):
        self.hooks = []
        named_modules = dict(self.model.named_modules())
        if layer_names is not None:
            for k in named_modules.keys():
                if k in layer_names:
                    hook = OutputHook()
                    named_modules[k].register_forward_hook(hook)
                    self.hooks.append(hook)
                    print(f"Attached output hook to layer {k}")

        elif layer_types is not None:
            for k in named_modules.keys():
                if type(named_modules[k]).__name__ in layer_types:
                    hook = OutputHook()
                    named_modules[k].register_forward_hook(hook)
                    self.hooks.append(hook)
                    print(f"Attached output hook to layer {k} of type {type(named_modules[k])}")

    def _update_best_model(self, value, epoch):
        '''Updates best metric and best model'''
        if self.reference_metric in ["train_loss", "wave_mse"]:
            if value <= self.best_metric:
                self.best_metric = value
                # We need to save/load state dict to untie self.model and self.best_model
                torch.save(self.model.state_dict(), self.best_model_state_dict_path)
                self.best_model.load_state_dict(torch.load(self.best_model_state_dict_path,
                                                           weights_only=True))
                self.best_epoch = epoch
        elif self.reference_metric in ["pesq", "stoi", "snr", "si-snr"]:
            if value >= self.best_metric:
                self.best_metric = value
                # We need to save/load state dict to untie self.model and self.best_model
                torch.save(self.model.state_dict(), self.best_model_state_dict_path)
                self.best_model.load_state_dict(torch.load(self.best_model_state_dict_path,
                                                           weights_only=True))
                self.best_epoch = epoch
        else:
            raise ValueError(f"reference metric must be in {self.header}")
