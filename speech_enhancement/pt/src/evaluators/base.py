# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

"""Base evaluator classes with all the boilerplate"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
import onnxruntime
import onnx

class BaseTorchEvaluator:
    '''Base class for torch model evaluator objects. 
    Handles boilerplate code common to all torch evaluators.'''
    def __init__(self,
                 model: nn.Module,
                 model_checkpoint: str,
                 eval_data: DataLoader,
                 metric_names: list,
                 logs_path: str,
                 device: str = "cuda:0",
                 device_memory_fraction: float = 0.5):
        '''
        Parameters
        ----------
        model, nn.Module : Torch model object to evaluate
        model_checkpoint, str or Path : Path to the state dict of the model to evaluate
        eval_data, Dataloader : Dataloader for the evaluation data.
        metric_names, list[str] : Name of the metrics the model is evaluated on
        logs_path, str or Path : Path to the directory where evaluation logs are to be saved
        device, str : Torch device on which to run the evaluation
        device_memory_fraction, float : Portion of memory used on device by the evaluator

        Notes
        -----
        Evaluators for specific torch models should inherit from this class, and 
        implement the _run_evaluation_step method, which should contain the evaluation logic
        , take as input a batch, and return a tuple of metrics (e.g. (mse, pesq, stoi, snr, si_snr))

        '''
        if device != "cpu":
            torch.cuda.set_per_process_memory_fraction(device_memory_fraction, device)

        self.model = model.to(device)
        if model_checkpoint is not None:
            model_state_dict = torch.load(model_checkpoint, map_location=device, weights_only=True)
            self.model.load_state_dict(model_state_dict)
        self.model.eval() # Just in case !
        self.eval_data = eval_data
        self.metric_names = metric_names
        self.logs_path = Path(logs_path)
        self.device = device

        # Dict containing average metrics computed on self.eval_data
        # Can also access self.metrics_dict.keys() for list of names of metrics
        self.metrics_dict = {}
        for key in self.metric_names:
            self.metrics_dict[key] = 0
        # Array contaning metrics for each sample
        # One row = one sample in eval_data
        self.metrics_array = np.zeros((len(eval_data), len(self.metrics_dict.keys())))

    def _run_evaluation_step(self, batch):
        raise NotImplementedError("No _run_evaluation_step method implemented"
            "BaseEvaluators should not be instantiated \n"
            "You should instead define your own Evaluator class that inherits\n"
            "from BaseTorchEvaluator and implements the _run_validation_step method.")

    def evaluate(self):
        '''Runs _run_evaluation_step until eval_data is exhausted, updates and saves metrics
           Saves a metrics_array containing metrics for each batch, and a metrics_dict containing
           average metrics over eval_data.'''
        # Create logs directory if it does not already exist
        self.logs_path.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            i = 0
            for batch in tqdm(self.eval_data):
                self.metrics_array[i:i+batch[0].shape[0]] = self._run_evaluation_step(batch)
                i+= batch[0].shape[0]
            for j, key in enumerate(self.metrics_dict.keys()):
                self.metrics_dict[key] = np.mean(self.metrics_array[:, j])
    
        # Save metrics dict and metrics array
        with open(self.logs_path / 'metrics_dict.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics_dict, f, ensure_ascii=False, indent=4)
        header = self.metrics_dict.keys()
        metrics_df = pd.DataFrame(self.metrics_array, columns=header)
        metrics_df.to_csv(self.logs_path / "detailed_metrics.csv", header=True)
        return self.metrics_dict, self.metrics_array
    
class BaseONNXEvaluator:
    '''Base class for ONNX model evaluator objects. 
    Handles boilerplate code common to all ONNX evaluators.'''
    def __init__(self,
                 eval_data: DataLoader,
                 metric_names: list,
                 logs_path: str,
                 model_path: str = None,
                 session: onnxruntime.InferenceSession = None):
        '''
        Parameters
        ----------
        model_path, str or Path : Path to the saved ONNX model to evaluate
        eval_data, Dataloader : Dataloader for the evaluation data.
        metric_names, list[str] : Name of the metrics the model is evaluated on
        logs_path, str or Path : Path to the directory where evaluation logs are to be saved
        session, onnxruntime InferenceSession : Session corresponding to the model to evaluate.
            Mutually exclusive with model_path.

        Notes
        -----
        Evaluators for specific ONNX models should inherit from this class, and 
        implement the _run_evaluation_step method, which should contain the evaluation logic
        , take as input a batch, and return a tuple of metrics (e.g. (mse, pesq, stoi, snr, si_snr))

        '''
        
        self.model_path = model_path
        # With the new API, this evaluator can get an InferenceSession as input
        if session is not None and model_path is not None:
            raise ValueError("session and model_path are mutually exclusive, but both were provided.")
        if session is not None:
            self.session = session
            self.model = onnx.load(self.session._model_path)
        else:
            self.model = onnx.load(self.model_path)
            self.session = onnxruntime.InferenceSession(self.model_path)
        self.eval_data = eval_data
        self.metric_names = metric_names
        self.logs_path = Path(logs_path)
        # Dict containing average metrics computed on self.eval_data
        # Can also access self.metrics_dict.keys() for list of names of metrics
        self.metrics_dict = {}
        for key in self.metric_names:
            self.metrics_dict[key] = 0
        # Array contaning metrics for each sample
        # One row = one sample in eval_data
        self.metrics_array = np.zeros((len(eval_data), len(self.metrics_dict.keys())))

        # Grab net input and net output nodes from the onnx graph
        # Use the in the subclasses or don't, it's there for convenience.
        self.output_nodes =[node.name for node in self.model.graph.output]
        self.input_all = [node.name for node in self.model.graph.input]
        self.input_initializer =  [node.name for node in self.model.graph.initializer]
        self.net_input_nodes = list(set(self.input_all)  - set(self.input_initializer))

    def _run_evaluation_step(self, batch):
        raise NotImplementedError("No _run_evaluation_step method implemented"
            "BaseEvaluators should not be instantiated \n"
            "You should instead define your own Evaluator class that inherits from BaseEvaluator and \n"
            "implements the _run_validation_step method.")

    def evaluate(self):
        # Create logs directory if it does not already exist
        self.logs_path.mkdir(parents=True, exist_ok=True)
        i = 0
        for batch in tqdm(self.eval_data):
            batch = list(map(lambda tensor: tensor.to("cpu").numpy(), batch))
            self.metrics_array[i:i+batch[0].shape[0]] = self._run_evaluation_step(batch)
            i+= batch[0].shape[0]
        for j, key in enumerate(self.metrics_dict.keys()):
            self.metrics_dict[key] = np.mean(self.metrics_array[:, j])
    
        # Save metrics dict and metrics array
        with open(self.logs_path / 'metrics_dict.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics_dict, f, ensure_ascii=False, indent=4)
        header = self.metrics_dict.keys()
        metrics_df = pd.DataFrame(self.metrics_array, columns=header)
        metrics_df.to_csv(self.logs_path / "detailed_metrics.csv", header=True)
        return self.metrics_dict, self.metrics_array