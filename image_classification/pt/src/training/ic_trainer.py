# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import logging
import os
import sys
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from timm import utils
from timm.data import (FastCollateMixup, Mixup,
                       resolve_data_config)
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)
from timm.loss import (BinaryCrossEntropy, JsdCrossEntropy,
                       LabelSmoothingCrossEntropy, SoftTargetCrossEntropy)
from timm.models import (load_checkpoint, model_parameters, resume_checkpoint,
                         safe_model_name)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common.model_utils.checkpoint_saver import CheckpointSaver
from common.onnx_utils.onnx_model_convertor import torch_model_export_static

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')

class ICTrainer:
    """
    Image Classification Trainer.

    This class encapsulates the training and validation logic for image
    classification models. It handles model initialization, data loading,
    training loop management, evaluation, and logging based on a provided
    configuration.

    Attributes:
        args (Namespace or dict): Configuration object containing training
            hyperparameters such as learning rate, number of epochs, optimizer
            type, checkpoint paths, logging options, and device selection.
        model (torch.nn.Module): The neural network model to train and evaluate.
        dataloaders (dict): A dictionary containing 'train' and 'val' 
            DataLoader objects for iterating over the dataset splits.

    Typical usage example:
        trainer = ICTrainer(args, model, dataloaders)
        trainer.train()
        trainer.evaluate()
    """

    def __init__(self, cfg, model, dataloaders):
        self.cfg = cfg
        from types import SimpleNamespace
        from common.utils import flatten_config
        # convert hydra heirarchical config to flat config
        args_raw = flatten_config(cfg)
        # convert dict based config to argument parser type
        if isinstance(args_raw, dict):
            args = SimpleNamespace(**args_raw)
        else:
            args = args_raw
        self.args = args
        self.args.input_size = self.args.input_shape
        self.model = model
        self.dataloaders = dataloaders
        # placeholders for components created later
        self.device = None
        self.use_amp = None
        self.amp_dtype = torch.float16
        self.amp_autocast = suppress
        self.loss_scaler = None

        self.optimizer = None
        self.model_ema = None

        self.loader_train = None
        self.loader_eval = None
        self.dataset_train = None

        self.train_loss_fn = None
        self.validate_loss_fn = None

        self.saver = None
        self.output_dir = os.path.join(args.output_dir, args.saved_models_dir)
        self.writer = None

        self.lr_scheduler = None
        self.num_epochs = None
        self.start_epoch = 0
        self.resume_epoch = None

        self.best_metric = None
        self.best_epoch = None

        self.mixup_fn = None
        self.collate_fn = None

    #-------------------- Train (main function)-----------------
    def train(self):
        self.setup_environment()
        self.process_model()
        self.create_optimizer()
        self.setup_amp_and_scaler()
        self.resume_if_needed()
        self.setup_model_ema()
        self.setup_distributed_and_compile()
        self.process_dataloaders()
        self.setup_losses()
        self.setup_checkpoint_and_logging()
        self.setup_lr_scheduler_and_start()
        self.train_loop()
        onnx_model = None
        if self.args.distributed:
            if self.args.local_rank == 0:
                onnx_model = torch_model_export_static(cfg=self.cfg, 
                                         model_dir=self.output_dir, 
                                         model=self.model.module.to("cpu"))
        else: 
            onnx_model = torch_model_export_static(cfg=self.cfg, 
                                         model_dir=self.output_dir, 
                                         model=self.model.to("cpu"))   
        return onnx_model

    # ------------------ Environment & Device ------------------
    def setup_environment(self):
        utils.setup_default_logging()

        args = self.args # copy by reference
        # cuda / cudnn settings
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # boolean flags and simple post-processing expected by downstream code
        args.prefetcher = not getattr(args, 'no_prefetcher', False)
        args.grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))

        self.device = args.device
        if args.distributed:
            _logger.info(
                'Training in distributed mode with multiple processes, 1 device per process.'
                f'Process {args.rank}, total {args.world_size}, device {args.device}.')
        else:
            _logger.info(f'Training with a single process on 1 device ({args.device}).')
        assert args.rank >= 0

        # resolve AMP arguments based on PyTorch / Apex availability
        self.use_amp = None
        self.amp_dtype = torch.float16
        if getattr(args, 'amp', False):
            if getattr(args, 'amp_impl', 'native') == 'apex':
                assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
                self.use_amp = 'apex'
                assert args.amp_dtype == 'float16'
            else:
                assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
                self.use_amp = 'native'
                assert args.amp_dtype in ('float16', 'bfloat16')
            if args.amp_dtype == 'bfloat16':
                self.amp_dtype = torch.bfloat16

        if getattr(args, 'fuser', None):
            utils.set_jit_fuser(args.fuser)
        if getattr(args, 'fast_norm', False):
            set_fast_norm()

    # ------------------ Model creation ------------------
    def process_model(self):
        args = self.args
        
        # optional head init
        if hasattr(self.model, 'get_classifier'):
            if getattr(args, 'head_init_scale', None) is not None:
                with torch.no_grad():
                    self.model.get_classifier().weight.mul_(args.head_init_scale)
                    self.model.get_classifier().bias.mul_(args.head_init_scale)
            if getattr(args, 'head_init_bias', None) is not None:
                nn.init.constant_(self.model.get_classifier().bias, args.head_init_bias)

        if getattr(args, 'num_classes', None) is None:
            if not hasattr(self.model, 'num_classes'):
                raise AssertionError('Model must have `num_classes` attr if not set on cmd line/config.')
            args.num_classes = self.model.num_classes

        if getattr(args, 'grad_checkpointing', False):
            # only call if model supports it
            try:
                self.model.set_grad_checkpointing(enable=True)
            except Exception:
                pass
        if utils.is_primary(args):
            _logger.info(
                f'Model {safe_model_name(args.model_name)} created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        # resolve data config for model
        data_config = resolve_data_config(vars(args), model=self.model, verbose=utils.is_primary(args))
        self.data_config = data_config

        # setup augmentation batch splits for contrastive loss or split bn
        num_aug_splits = 0
        if getattr(args, 'aug_splits', 0) > 0:
            assert args.aug_splits > 1, 'A split of 1 makes no sense'
            num_aug_splits = args.aug_splits
        self.num_aug_splits = num_aug_splits

        # enable split bn (separate bn stats per batch-portion)
        if getattr(args, 'split_bn', False):
            assert num_aug_splits > 1 or getattr(args, 'resplit', False)
            self.model = convert_splitbn_model(self.model, max(num_aug_splits, 2))

        # move model to device, enable channels last layout if set
        self.model.to(device=self.device)
        if getattr(args, 'channels_last', False):
            self.model.to(memory_format=torch.channels_last)

        # setup synchronized BatchNorm for distributed training
        if args.distributed and getattr(args, 'sync_bn', False):
            args.dist_bn = ''  # disable dist_bn when sync BN active
            assert not getattr(args, 'split_bn', False)
            if has_apex and self.use_amp == 'apex':
                # Apex SyncBN used with Apex AMP
                # WARNING this won't currently work with models using BatchNormAct2d
                self.model = convert_syncbn_model(self.model)
            else:
                self.model = convert_sync_batchnorm(self.model)
            if utils.is_primary(args):
                _logger.info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        # torchscript checks
        if getattr(args, 'torchscript', False):
            assert not getattr(args, 'torchcompile', False)
            assert not self.use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not getattr(args, 'sync_bn', False), 'Cannot use SyncBatchNorm with torchscripted model'
            self.model = torch.jit.script(self.model)

        if not args.lr:
            global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
            batch_ratio = global_batch_size / args.lr_base_size
            if not args.lr_base_scale:
                on = args.opt.lower()
                args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
            if args.lr_base_scale == 'sqrt':
                batch_ratio = batch_ratio ** 0.5
            args.lr = args.lr_base * batch_ratio
            if utils.is_primary(args):
                _logger.info(
                    f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                    f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    # ------------------ Optimizer & AMP ------------------
    def create_optimizer(self):
        args = self.args
        self.optimizer = create_optimizer_v2(
            self.model,
            **optimizer_kwargs(cfg=args),
            **(getattr(args, 'opt_kwargs', {}) or {}),
        )

    def setup_amp_and_scaler(self):
        args = self.args
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # default: no-op
        self.loss_scaler = None
        if self.use_amp == 'apex':
            assert self.device.type == 'cuda'
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.loss_scaler = ApexScaler()
            if utils.is_primary(args):
                _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif self.use_amp == 'native':
            try:
                self.amp_autocast = partial(torch.autocast, device_type=self.device.type, dtype=self.amp_dtype)
            except (AttributeError, TypeError):
                # fallback to CUDA only AMP for PyTorch < 1.10
                assert self.device.type == 'cuda'
                self.amp_autocast = torch.cuda.amp.autocast
            if self.device.type == 'cuda' and self.amp_dtype == torch.float16:
                # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
                self.loss_scaler = NativeScaler()
            if utils.is_primary(args):
                _logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            if utils.is_primary(args):
                _logger.info('AMP not enabled. Training in float32.')

    # ------------------ Resume & EMA & DDP & Compile ------------------
    def resume_if_needed(self):
        args = self.args
        self.resume_epoch = None
        if getattr(args, 'resume_training_from', None):
            print("Resuming training from: ", args.resume_training_from)
            self.resume_epoch = resume_checkpoint(
                self.model,
                args.resume_training_from,
                optimizer=None if getattr(args, 'no_resume_opt', False) else self.optimizer,
                loss_scaler=None if getattr(args, 'no_resume_opt', False) else self.loss_scaler,
                log_info=utils.is_primary(args),
            )
            print("Resuming training from: ", args.resume_training_from)
            print("Resuming training from epoch: ", self.resume_epoch)
    
    def setup_model_ema(self):
        args = self.args
        self.model_ema = None
        if getattr(args, 'model_ema', False):
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
            self.model_ema = utils.ModelEmaV2(
                self.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
            if getattr(args, 'resume_training_from', None):
                # if a resume_training_from path provided, try to load EMA weights from it
                load_checkpoint(self.model_ema.module, args.resume_training_from, use_ema=True)

    def setup_distributed_and_compile(self):
        args = self.args
        if args.distributed:
            if has_apex and self.use_amp == 'apex':
                if utils.is_primary(args):
                    _logger.info("Using NVIDIA APEX DistributedDataParallel.")
                self.model = ApexDDP(self.model, delay_allreduce=True)
            else:
                if utils.is_primary(args):
                    _logger.info("Using native Torch DistributedDataParallel.")
                # device variable may be device id / torch.device depending on utils.init_distributed_device
                self.model = NativeDDP(self.model, device_ids=[self.device], broadcast_buffers=not getattr(args, 'no_ddp_bb', False))
            # NOTE: EMA model does not need to be wrapped by DDP

        if getattr(args, 'torchcompile', False):
            assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
            self.model = torch.compile(self.model, backend=args.torchcompile)

    # ------------------ Data loaders ------------------
    def process_dataloaders(self):
        args = self.args
        # setup mixup / cutmix
        self.collate_fn = None
        self.mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or getattr(args, 'cutmix_minmax', None) is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=getattr(args, 'cutmix_minmax', None),
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.num_classes
            )
            if args.prefetcher:
                assert not self.num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                self.collate_fn = FastCollateMixup(**mixup_args)
            else:
                self.mixup_fn = Mixup(**mixup_args)

        self.loader_train, self.loader_eval = self.dataloaders['train'], self.dataloaders['valid']
        self.dataset_train = self.loader_train.dataset

    # ------------------ Loss ------------------
    def setup_losses(self):
        args = self.args
        num_aug_splits = self.num_aug_splits
        mixup_active = self.mixup_fn is not None or (self.collate_fn is not None)

        if getattr(args, 'jsd_loss', False):
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            self.train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if getattr(args, 'bce_loss', False):
                self.train_loss_fn = BinaryCrossEntropy(target_threshold=getattr(args, 'bce_target_thresh', None))
            else:
                self.train_loss_fn = SoftTargetCrossEntropy()
        elif getattr(args, 'smoothing', 0):
            if getattr(args, 'bce_loss', False):
                self.train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=getattr(args, 'bce_target_thresh', None))
            else:
                self.train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            self.train_loss_fn = nn.CrossEntropyLoss()
        self.train_loss_fn = self.train_loss_fn.to(device=self.device)
        self.validate_loss_fn = nn.CrossEntropyLoss().to(device=self.device)

    # ------------------ Checkpoint, logging, saver ------------------
    def setup_checkpoint_and_logging(self):
        args = self.args
        # setup checkpoint saver and eval metric tracking
        eval_metric = getattr(args, 'eval_metric', 'accuracy')
        self.best_metric = None
        self.best_epoch = None
        self.saver = None
        self.output_dir = os.path.join(args.output_dir, args.saved_models_dir)

        if utils.is_primary(args):
            if getattr(args, 'project_name', None):
                exp_name = args.project_name
            else:
                exp_name = '-'.join([
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model_name),
                    str(self.data_config['input_size'][-1])
                ])
            decreasing = True if eval_metric == 'loss' else False
            self.saver = CheckpointSaver(
                model=self.model,
                optimizer=self.optimizer,
                args=args,
                model_ema=self.model_ema,
                amp_scaler=self.loss_scaler,
                checkpoint_dir=self.output_dir,
                recovery_dir=self.output_dir,
                decreasing=decreasing,
                max_history=getattr(args, 'checkpoint_hist', 10)
            )
            # write args to file # different from before
            args_text = yaml.safe_dump(vars(args))
            with open(os.path.join(self.output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)
            if getattr(args, 'log_tb', False):
                self.writer = SummaryWriter(log_dir=self.output_dir)

        if utils.is_primary(args) and getattr(args, 'log_wandb', False):
            if has_wandb:
                wandb.init(project=getattr(args, 'experiment', None), config=args)
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

    # ------------------ LR scheduler and start epoch ------------------
    def setup_lr_scheduler_and_start(self):
        args = self.args
        updates_per_epoch = (len(self.loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
        self.lr_scheduler, self.num_epochs = create_scheduler_v2(
            self.optimizer,
            **scheduler_kwargs(args),
            updates_per_epoch=updates_per_epoch,
        )
        self.start_epoch = 0
        if getattr(args, 'start_epoch', None) is not None:
            self.start_epoch = args.start_epoch
        elif self.resume_epoch is not None:
            self.start_epoch = self.resume_epoch
        if self.lr_scheduler is not None and self.start_epoch > 0:
            if getattr(args, 'sched_on_updates', False):
                self.lr_scheduler.step_update(self.start_epoch * updates_per_epoch)
            else:
                self.lr_scheduler.step(self.start_epoch)

        if utils.is_primary(args):
            _logger.info(
                f'Scheduled epochs: {self.num_epochs}. LR stepped per {"epoch" if self.lr_scheduler.t_in_epochs else "update"}.')

    # ------------------ Training loop ------------------
    def train_loop(self):
        args = self.args
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                # set epoch for samplers if present
                if hasattr(self.dataset_train, 'set_epoch'):
                    self.dataset_train.set_epoch(epoch)
                elif args.distributed and hasattr(self.loader_train.sampler, 'set_epoch'):
                    self.loader_train.sampler.set_epoch(epoch)

                train_metrics = self.train_one_epoch(
                    epoch,
                    self.model,
                    self.loader_train,
                    self.optimizer,
                    self.train_loss_fn,
                    args,
                    lr_scheduler=self.lr_scheduler,
                    saver=self.saver,
                    output_dir=self.output_dir,
                    amp_autocast=self.amp_autocast,
                    loss_scaler=self.loss_scaler,
                    model_ema=self.model_ema,
                    mixup_fn=self.mixup_fn
                )

                # distributed bn sync if requested
                if args.distributed and getattr(args, 'dist_bn', None) in ('broadcast', 'reduce'):
                    if utils.is_primary(args):
                        _logger.info("Distributing BatchNorm running means and vars")
                    utils.distribute_bn(self.model, args.world_size, args.dist_bn == 'reduce')

                eval_metrics = self.validate(
                    self.model,
                    self.loader_eval,
                    self.validate_loss_fn,
                    args,
                    amp_autocast=self.amp_autocast,
                )
                eval_metrics_unite = eval_metrics
                ema_eval_metrics = None

                if self.model_ema is not None and not getattr(args, 'model_ema_force_cpu', False):
                    if args.distributed and getattr(args, 'dist_bn', None) in ('broadcast', 'reduce'):
                        utils.distribute_bn(self.model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = self.validate(
                        self.model_ema.module,
                        self.loader_eval,
                        self.validate_loss_fn,
                        args,
                        amp_autocast=self.amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    # choose the best model between normal and EMA using eval_metric
                    eval_metric = getattr(args, 'eval_metric', 'top1')
                    # in some configs metric is 'loss' where smaller is better; original code compared using >
                    if ema_eval_metrics[eval_metric] > eval_metrics[eval_metric]:
                        eval_metrics_unite = ema_eval_metrics

                if getattr(args, 'dryrun', False):
                    break

                if self.output_dir is not None:
                    lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                    if getattr(args, 'log_tb', False) and self.writer is not None:
                        for key, value in train_metrics.items():
                            self.writer.add_scalar('train/' + key, value, epoch)
                        for key, value in eval_metrics_unite.items():
                            self.writer.add_scalar('eval/' + key, value, epoch)
                        for i, lr in enumerate(lrs):
                            self.writer.add_scalar(f'lr/{i}', lr, epoch)
                    utils.update_summary(
                        epoch,
                        train_metrics,
                        eval_metrics_unite,
                        filename=os.path.join(self.output_dir, 'summary.csv'),
                        lr=sum(lrs) / len(lrs),
                        write_header=self.best_metric is None,
                        log_wandb=getattr(args, 'log_wandb', False) and has_wandb,
                    )

                if self.saver is not None:
                    # save proper checkpoint with eval metric
                    eval_metric = getattr(args, 'eval_metric', 'top1')
                    save_metric = eval_metrics.get(eval_metric, None)
                    if ema_eval_metrics:
                        save_metric_ema = ema_eval_metrics.get(eval_metric, -1)
                    else:
                        save_metric_ema = -1
                    self.best_metric, self.best_epoch = self.saver.save_checkpoint(epoch, metric=save_metric, metric_ema=save_metric_ema)

                if self.lr_scheduler is not None:
                    # step LR for next epoch
                    self.lr_scheduler.step(epoch + 1, eval_metrics_unite.get(getattr(args, 'eval_metric', 'top1'), 0))

        except KeyboardInterrupt:
            pass

        if self.best_metric is not None:
            _logger.info('*** Best metric: {0} (epoch {1})'.format(self.best_metric, self.best_epoch))

    # ------------------ train_one_epoch (member) ------------------
    def train_one_epoch(
            self,
            epoch,
            model,
            loader,
            optimizer,
            loss_fn,
            args=None,
            device=None,
            lr_scheduler=None,
            saver=None,
            output_dir=None,
            amp_autocast=suppress,
            loss_scaler=None,
            model_ema=None,
            mixup_fn=None,
            model_kd=None,
    ):
        # This function is intentionally left functionally the same as the original,
        # but accepts optional args/device and falls back to trainer attributes.
        if args is None:
            args = self.args
        if device is None:
            device = self.device if self.device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        if getattr(args, 'mixup_off_epoch', None) and epoch >= getattr(args, 'mixup_off_epoch'):
            if getattr(args, 'prefetcher', False) and hasattr(loader, 'mixup_enabled') and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        has_no_sync = hasattr(model, "no_sync")
        update_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()

        model.train()

        accum_steps = args.grad_accum_steps
        last_accum_steps = len(loader) % accum_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1
        last_batch_idx_to_accum = len(loader) - last_accum_steps

        data_start_time = update_start_time = time.time()
        optimizer.zero_grad()
        update_sample_count = 0
        #max_debug_steps = 4 #debug
        for batch_idx, (input, target) in enumerate(loader):
            #if batch_idx >= max_debug_steps: #debug
            #    break #debug
            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps
            if batch_idx >= last_batch_idx_to_accum:
                accum_steps = last_accum_steps

            if not getattr(args, 'prefetcher', False):
                input, target = input.to(device), target.to(device)
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            if getattr(args, 'channels_last', False):
                input = input.contiguous(memory_format=torch.channels_last)

            # multiply by accum steps to get equivalent for full update
            data_time_m.update(accum_steps * (time.time() - data_start_time))

            def _forward():
                with amp_autocast():
                    output = model(input)
                    loss = loss_fn(output, target)

                if accum_steps > 1:
                    loss /= accum_steps
                return loss

            def _backward(_loss):
                if loss_scaler is not None:
                    loss_scaler(
                        _loss,
                        optimizer,
                        clip_grad=getattr(args, 'clip_grad', None),
                        clip_mode=getattr(args, 'clip_mode', None),
                        parameters=model_parameters(model, exclude_head='agc' in getattr(args, 'clip_mode', '')),  # original logic
                        create_graph=second_order,
                        need_update=need_update,
                    )
                else:
                    _loss.backward(create_graph=second_order)
                    if need_update:
                        if getattr(args, 'clip_grad', None) is not None:
                            utils.dispatch_clip_grad(
                                model_parameters(model, exclude_head='agc' in getattr(args, 'clip_mode', '')),
                                value=args.clip_grad,
                                mode=getattr(args, 'clip_mode', None),
                            )
                        optimizer.step()

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss = _forward()
                    _backward(loss)
            else:
                loss = _forward()
                _backward(loss)

            if not getattr(args, 'distributed', False):
                losses_m.update(loss.item() * accum_steps, input.size(0))
            update_sample_count += input.size(0)

            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

            if getattr(args, 'synchronize_step', False) and device.type == 'cuda':
                torch.cuda.synchronize()
            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % getattr(args, 'log_interval', 10) == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if getattr(args, 'distributed', False):
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                    update_sample_count *= args.world_size

                if utils.is_primary(args):
                    _logger.info(
                        f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                        f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )

                    if getattr(args, 'save_images', False) and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True
                        )

            if getattr(args, 'dryrun', False):
                break

            if saver is not None and getattr(args, 'recovery_interval', None) and (
                    (update_idx + 1) % getattr(args, 'recovery_interval') == 0):
                saver.save_recovery(epoch, batch_idx=update_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            update_sample_count = 0
            data_start_time = time.time()
            # end for

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])

    # ------------------ validate (member) ------------------
    def validate(self,
                 model,
                 loader,
                 loss_fn,
                 args=None,
                 device=None,
                 amp_autocast=suppress,
                 log_suffix=''):
        if args is None:
            args = self.args
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        batch_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()
        top1_m = utils.AverageMeter()
        top5_m = utils.AverageMeter()

        model.eval()

        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not getattr(args, 'prefetcher', False):
                    input = input.to(device)
                    target = target.to(device)
                if getattr(args, 'channels_last', False):
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # augmentation reduction
                    reduce_factor = getattr(args, 'tta', 1)
                    if reduce_factor > 1:
                        output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                        target = target[0:target.size(0):reduce_factor]

                    loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                if getattr(args, 'distributed', False):
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    acc1 = utils.reduce_tensor(acc1, args.world_size)
                    acc5 = utils.reduce_tensor(acc5, args.world_size)
                else:
                    reduced_loss = loss.data

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if utils.is_primary(args) and (last_batch or batch_idx % getattr(args, 'log_interval', 10) == 0):
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                        f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                        f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                        f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                        f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                    )
                if getattr(args, 'dryrun', False):
                    break

        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

        return metrics



