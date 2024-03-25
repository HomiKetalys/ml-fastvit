#
# For acknowledgement see accompanying ACKNOWLEDGEMENTS file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

import yaml
from timm.layers import apply_test_time_pool
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    RealLabelsImagenet,
)
from timm.utils import (
    accuracy,
    AverageMeter,
    natural_key,
    setup_default_logging,
    set_jit_legacy,
)

import models
from common_utils.utils import tfOrtModelRuner
from models.modules.mobileone import reparameterize_model

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("validate")

parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("--data_dir", default="", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--split",
    metavar="NAME",
    default="validation",
    help="dataset split (default: validation)",
)
parser.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="NAME",
    default="dpn92",
    help="model architecture (default: dpn92)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--input-size",
    default=[3, 256, 256],
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 256 256), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop pct",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--num-classes", type=int, default=None, help="Number classes in dataset"
)
parser.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--log-freq",
    default=10,
    type=int,
    metavar="N",
    help="batch logging frequency (default: 10)",
)
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--tf-preprocessing",
    action="store_true",
    default=False,
    help="Use Tensorflow preprocessing pipeline (require CPU TF installed",
)
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    help="use ema version of weights if present",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--legacy-jit",
    dest="legacy_jit",
    action="store_true",
    help="use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance",
)
parser.add_argument(
    "--results-file",
    default="",
    type=str,
    metavar="FILENAME",
    help="Output csv file for validation results (summary)",
)
parser.add_argument(
    "--real-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Real labels JSON file for imagenet evaluation",
)
parser.add_argument(
    "--valid-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Valid label indices txt file for validation of partial label space",
)
parser.add_argument(
    "--use-inference-mode",
    dest="use_inference_mode",
    action="store_true",
    default=False,
    help="use inference mode version of model definition.",
)

activations = {
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
}


class ClassifierTfOrt():
    def __init__(self, cfg, model_root):
        super(ClassifierTfOrt, self).__init__()
        name_list = os.listdir(model_root)
        model_front_path = None
        model_post_path = None
        model_path = None
        self.separation = cfg["separation"]
        self.separation_scale = cfg["separation_scale"]
        for name in name_list:
            if name.endswith(".tflite") or name.endswith(".onnx"):
                if "front" in name:
                    model_front_path = os.path.join(model_root, name)
                elif "post" in name:
                    model_post_path = os.path.join(model_root, name)
                else:
                    model_path = os.path.join(model_root, name)
        if model_path is None:
            assert os.path.splitext(model_front_path)[-1] == os.path.splitext(model_post_path)[-1]
            self.model_type = os.path.splitext(model_front_path)[-1]
            self.model_front = tfOrtModelRuner(model_front_path)
            self.model_post = tfOrtModelRuner(model_post_path)
            if model_front_path.endswith(".tflite"):
                std0, mean0 = self.model_front.model_output_details[0]["quantization"]
                std1, mean1 = self.model_post.model_input_details["quantization"]
                self.fix0 = std0 / std1
                self.fix1 = -self.fix0 * mean0 + mean1
            self.sp = 1
            self.weight, self.bias = self.model_front.model_input_details["quantization"]
            self.input_size=self.separation_scale*self.model_front.model_input_details["shape"][1:]
        else:
            self.model = tfOrtModelRuner(model_path)
            self.sp = 0
            self.model_type = os.path.splitext(model_path)[-1]
            self.weight, self.bias = self.model.model_input_details["quantization"]
            self.input_size = self.model.model_input_details["shape"][1:]

    def __call__(self, inputs):
        pred_list0 = []
        for x in inputs:
            if self.model_type == ".tflite":
                x = np.clip(x.permute(1, 2, 0).cpu().numpy() / self.weight + self.bias, -128, 127)
                x = x.astype("int8")
                h, w, c = x.shape[:3]
            else:
                x = x.cpu().numpy()
                c, h, w = x.shape[:3]
            h0 = h
            w0 = w
            if self.sp == 1:
                y_list = []
                for r in range(0, self.separation_scale):
                    for c in range(0, self.separation_scale):
                        if self.model_type == ".tflite":
                            x_ = x[None, r * h // self.separation_scale:(r + 1) * h // self.separation_scale,
                                 c * w // self.separation_scale:(c + 1) * w // self.separation_scale, :]
                        else:
                            x_ = x[None, :, r * h // self.separation_scale:(r + 1) * h // self.separation_scale,
                                 c * w // self.separation_scale:(c + 1) * w // self.separation_scale]
                        y = self.model_front(x_)[0]
                        y_list.append(y)

                if self.model_type == ".tflite":
                    h, w, c = y_list[0].shape[:3]
                    y = np.zeros((h * self.separation_scale, w * self.separation_scale, c), dtype="int8")
                else:
                    c, h, w = y_list[0].shape[:3]
                    y = np.zeros((c, h * self.separation_scale, w * self.separation_scale), dtype="float32")
                id = 0
                for r in range(0, self.separation_scale):
                    for c in range(0, self.separation_scale):
                        if self.model_type == ".tflite":
                            y[r * h:(r + 1) * h, c * w:(c + 1) * w, :] = y_list[id]
                        else:
                            y[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = y_list[id]
                        id += 1

                y = y[None, :, :, :].astype('float32')
                if self.model_type == ".tflite":
                    y = np.clip(y * self.fix0 + self.fix1, -128, 127)
                    y = y.astype('int8')
                out = self.model_post(y)

            else:
                out = self.model(x[None, :, :, :])
            out0 = torch.tensor(out, device=inputs.device)
            pred_list0.append(out0)
        out0 = torch.cat(pred_list0, dim=0)
        return out0


def cfg_analyzer(cfg_path, model_path, eval_type,img_size=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args()
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript,
        inference_mode=args.use_inference_mode,
        activation=activations[args.global_act],
        separation=args.separation,
        separation_scale=args.separation_scale,
    )

    if eval_type == 0:
        load_checkpoint(model, model_path, args.use_ema)

    # Reparameterize model
    model.eval()
    if not args.use_inference_mode:
        _logger.info("Reparameterizing Model %s" % (args.model))
        model = reparameterize_model(model)
    # setattr(model, "pretrained_cfg", model.__dict__["default_cfg"])

    data_config = resolve_data_config(
        vars(args), model=model, use_test_size=True, verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(
            model, data_config, use_test_size=True
        )

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
    )

    if args.valid_labels:
        with open(args.valid_labels, "r") as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(
            dataset.filenames(basename=True), real_json=args.real_labels
        )
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    model.eval()
    model.cuda()
    if eval_type != 0:
        model = ClassifierTfOrt(cfg, model_path)
        data_config["input_size"] = (data_config["input_size"][0], model.input_size[0], model.input_size[1])
    elif img_size is not None:
        data_config["input_size"]=(data_config["input_size"][0],img_size[0],img_size[1])
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    return model, args, data_config, amp_autocast, valid_labels, criterion, real_labels, batch_time, loader, losses, top1, top5, crop_pct


def validate(cfg_path, model_path, eval_type,img_size=None):
    model, args, data_config, amp_autocast, valid_labels, criterion, real_labels, batch_time, loader, losses, top1, top5, crop_pct = cfg_analyzer(
        cfg_path, model_path, eval_type,img_size)
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn(
            (args.batch_size,) + tuple(data_config["input_size"])
        ).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                print(
                    "Test: [{0:>4d}/{1}]  "
                    "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                    "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        img_size=data_config["input_size"][-1],
        cropt_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )

    print(
        " * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})".format(
            results["top1"], results["top1_err"], results["top5"], results["top5_err"]
        )
    )
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + "/*.pth.tar")
        checkpoints += glob.glob(args.checkpoint + "/*.pth")
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == "all":
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True, exclude_filters=["*_in21k", "*_in22k"]
            )
            model_cfgs = [(n, "") for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, "") for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or "./results-all.csv"
        _logger.info(
            "Running bulk validation on these pretrained models: {}".format(
                ", ".join(model_names)
            )
        )
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print("Validating with batch size: %d" % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print(
                                "Validation failed with no ability to reduce batch size. Exiting."
                            )
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result["checkpoint"] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x["top1"], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode="w") as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == "__main__":
    main()
