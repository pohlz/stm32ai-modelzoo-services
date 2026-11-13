# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType
from speech_enhancement.pt.src.quantization import DataLoaderDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed
from common.quantization import define_extra_options
from onnxruntime import InferenceSession

class SEONNXPTQQuantizer:
    '''Post-training quantizer for ONNX speech enhancement models.

    Notes
    -----
    Wraps ONNX Runtime static quantization (QDQ, INT8) for single-input models.
    Uses a PyTorch `DataLoader` via `DataLoaderDataReader` for calibration data.
    '''
    def __init__(self, cfg, model, dataloaders):
        '''Initialize the ONNX PTQ quantizer.

        Parameters
        ----------
        cfg, object : User configuration with quantization parameters and output directory settings.
        model, onnxruntime.InferenceSession : Float ONNX model session to be quantized.
        dataloaders, dict[str, torch.utils.data.DataLoader] : Contains `quant_dl` for calibration samples (batch size 1).

        Notes
        -----
        - Expects single-input ONNX models; multi-input is not supported.
        '''

        self.cfg = cfg
        self.model = model
        self.quant_dl = dataloaders["quant_dl"]

        # Load op types to quantize, calibration method and extra options
        self.op_types_to_quantize = cfg.quantization.onnx_quant_parameters.op_types_to_quantize
        assert (isinstance(self.op_types_to_quantize, list) or self.op_types_to_quantize is None), "op_types_to_quantize must be a list of str or None"
        
        self.calibrate_method = getattr(CalibrationMethod, cfg.quantization.onnx_quant_parameters.calibrate_method)
        self.extra_options = define_extra_options(cfg=cfg)
        self.output_dir = Path(cfg.output_dir, cfg.general.saved_models_dir)
        self.output_dir.mkdir(exist_ok=True)

        # self.model is passed by the get_model API and should be an onnxruntime.InferenceSession
        # We need the actual model to infer the name of input/output nodes and perform some checks
        self.float_onnx_model = onnx.load(self.model._model_path)

        # Get input/output node names
        # Output node name is unused for now
        output_nodes =[node.name for node in self.float_onnx_model.graph.output]
        input_all = [node.name for node in self.float_onnx_model.graph.input]
        input_initializer =  [node.name for node in self.float_onnx_model.graph.initializer]
        net_input_nodes = list(set(input_all)  - set(input_initializer))

        # print(f"Input node(s) : {net_input_nodes}")
        # print(f"Output node(s) : {output_nodes}")

        # If model has multiple inputs, throw an unsupported error
        # Will add support for multi-input models w/ decomposed LSTM support
        if len(net_input_nodes) > 1:
            raise NotImplementedError("Multi-input models are currently unsupported for quantization in the zoo")

        self.data_reader = DataLoaderDataReader(quant_dl=self.quant_dl,
                                            input_name=net_input_nodes[0],
                                            replace_dl_collate=False)
        
    def quantize(self):
        '''Run ONNX quantization preprocessing, quantization, and optional static-shape conversion.

        Notes
        -----
        - Preprocesses the float ONNX model with `quant_pre_process` for shape inference.
        - Quantizes with `quantize_static` using QDQ format and INT8 weights/activations.
        - Cleans up opset imports to match the original model.
        - Fixes a dynamic sequence axis to a static value and repairs output shapes.

        Returns
        -------
        tuple : (`quantized_model_session`, `quantized_static_model_session`)
            - `quantized_model_session`, onnxruntime.InferenceSession : Quantized ONNX with dynamic input shape.
            - `quantized_static_model_session`, onnxruntime.InferenceSession : Quantized ONNX with fixed sequence length.
        '''

        # Preprocess the float model
        
        onnx_prep_path = Path(self.output_dir, "preprocessed_model.onnx") 
        
        quant_pre_process(input_model=self.model._model_path, output_model_path=onnx_prep_path)
        print(f"[INFO] Saved preprocessed float ONNX model at {onnx_prep_path}")

        onnx_prep_model = onnx.load(onnx_prep_path)

        # Get original model's opsets
        orig_opsets = self.float_onnx_model.opset_import
        # Remove superfluous ONNX opsets added by the preprocessing step
        del onnx_prep_model.opset_import[:]
        for op in orig_opsets:
            opset = onnx_prep_model.opset_import.add()
            opset.domain = op.domain
            opset.version = op.version

        print("Opset imports after cleanup :")
        for opset in onnx_prep_model.opset_import:
            print("opset domain=%r version=%r" % (opset.domain, opset.version))

        onnx.save(onnx_prep_model, onnx_prep_path)

        # Call quantize_static

        quantized_model_path = Path(self.output_dir, "quantized_model_int8.onnx")
        per_channel = self.cfg.quantization.granularity == 'per_channel'
        quantize_static(onnx_prep_path,
                        quantized_model_path,
                        self.data_reader,
                        op_types_to_quantize=self.op_types_to_quantize, # e.g. ["Conv", "LSTM"]
                        calibrate_method=self.calibrate_method, 
                        quant_format=QuantFormat.QDQ,
                        per_channel=per_channel,
                        weight_type=QuantType.QInt8,
                        activation_type = QuantType.QInt8,
                        reduce_range=self.cfg.quantization.reduce_range,
                        extra_options=self.extra_options) # Add extra options here
        
        quantized_static_model_path = Path(self.output_dir, "quantized_model_int8_static.onnx")
        quant_model = onnx.load(quantized_model_path)

        make_dim_param_fixed(quant_model.graph,
                             param_name=self.cfg.quantization.static_axis_name,
                             value=self.cfg.quantization.static_sequence_length)
        fix_output_shapes(quant_model)
        onnx.save(quant_model, quantized_static_model_path)

        print("[INFO] Successfully converted quantized model to static input shape")

        print("\n [INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")


        quantized_model_session = InferenceSession(quantized_model_path)
        quantized_static_model_session = InferenceSession(quantized_static_model_path)
        return quantized_model_session, quantized_static_model_session
        