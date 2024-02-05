import os
import time
import re
from typing import Mapping, Any
from copy import deepcopy
import unittest
import warnings

import torch
import torch.nn as nn
from torch.nn.utils import prune

import snntorch as snn
from snntorch import utils
from snntorch import surrogate


def new_activation(spiking=False, beta=0.5, spike_grad=None, threshold=1.):
    if spiking:
        return snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=threshold)
    else:
        return nn.ReLU(inplace=True)

def new_stage(block, in_channels, out_channels, num_blocks, stride=2, padding=1,
              spiking=False, beta=0.5, spike_grad=None, threshold=1.):
    strides = [stride] + [1] * (num_blocks-1)
    blocks = nn.Sequential()
    current_channels = in_channels
    for i, stride in enumerate(strides):
        blk = block(current_channels, out_channels, stride, padding,
                    spiking, beta, spike_grad, threshold)
        blocks.add_module(f'block{i+1}', blk)
        current_channels = out_channels
    return blocks

class Classifier(nn.Module):
    def __init__(self, in_modules, out_classes, cost=0., pool=4):
        """
        :param in_modules: Number of input modules
        :param out_classes: Number of output classes
        :param cost: Dropout factor as cost of the network for using that classificator
        """
        super(Classifier, self).__init__()

        self.dequant = torch.ao.quantization.DeQuantStub()

        self.avg_pool = nn.AvgPool2d(pool)
        self.dropout = nn.Dropout(p=1-cost)
        self.linear = nn.Linear(in_modules, out_classes)
        self.softmax = nn.Softmax(dim=1)

        self.stopped = 0

    def reset(self):
        self.stopped = 0

    def forward(self, x, stop_threshold=0.75):
        out = self.avg_pool(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.dequant(out)
        softmax_out = nn.Softmax(dim=1)(out)
        stopped = False
        if softmax_out.shape[0] == 1:
            best = softmax_out.max(dim=1)[0]
            stopped = float(best) >= stop_threshold
            self.stopped += stopped
        # else:
        #     stops = softmax_out.max(dim=1)[0] >= stop_threshold
        #     self.stopped += torch.sum(stops).item()
        return out, stopped

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1,
                 spiking=False, beta=0.5, spike_grad=None, threshold=1.):
        super(ResBlock, self).__init__()

        self.spiking = spiking

        # Quantization stubs
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = new_activation(spiking, beta, spike_grad, threshold)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.expansion = 1
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut.add_module(f'conv', nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1,
                                                        stride=stride, bias=False))
            self.shortcut.add_module(f'bn', nn.BatchNorm2d(self.expansion * out_channels))
        self.act2 = new_activation(spiking, beta, spike_grad, threshold)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if self.spiking:
            out = self.dequant(out)
        out = self.act1(out)
        if self.spiking:
            out = self.quant(out)
        out = self.conv2(out)
        out = self.bn2(out)
        sct = self.shortcut(x)
        out = self.dequant(out)
        sct = self.dequant(sct)
        out += sct
        if not self.spiking:
            out = self.quant(out)
        out = self.act2(out)
        if self.spiking:
            out = self.quant(out)
        return out

    def spike_reset(self):
        utils.reset(self)

class ResNet(nn.Module):
    def __init__(self, in_shape, out_classes=10, initial_channels=64, stage_channels=None,
                 num_blocks_per_stage=2, initial_stride=1, stage_stride=2, padding=1,
                 spiking=False, beta=0.5, surrogate_alpha=2.0, threshold=1., neglect=0., *args, **kwargs):
        super(ResNet, self).__init__()
        self.device = 'cpu'
        self.in_shape = in_shape
        self.out_classes = out_classes
        if stage_channels is None:
            stage_channels = [64, 128, 256, 512]
        self.config = {'in_shape': in_shape, 'out_classes': out_classes, 'initial_channels': initial_channels,
                       'stage_channels':stage_channels, 'num_blocks_per_stage': num_blocks_per_stage,
                       'initial_stride': initial_stride, 'stage_stride': stage_stride, 'padding': padding,
                       'spiking': spiking, 'surrogate_alpha': surrogate_alpha, 'threshold': threshold,
                       'neglect': neglect}

        # Quantization stubs
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.qconfig = None

        # State
        self.spiking = spiking
        self.neglect = True if neglect > 0 else False
        self.fused = False
        self.quant_prepare = False
        self.quant_aware = False
        self.quantized = False
        self.pruned = False
        self.prune_arg = ([], {})
        self.quant_sparsity = 0.
        self.quant_params = 0
        self.quant_avg_weight = 0.

        # Hyperparameters
        final_channels = stage_channels[-1]
        num_stages = len(stage_channels)
        self.hyperparameters = {
            'in_features': in_shape[1],
            'in_samples': in_shape[2],
            'in_channels': in_shape[0],
            'out_classes': out_classes,
            'initial_channels': initial_channels,
            'final_channels': final_channels,
            'stage_channels': stage_channels,
            'initial_stride': initial_stride,
            'stage_stride': stage_stride,
            'padding': padding,
            'num_stages': num_stages,
            'num_blocks_per_channel': num_blocks_per_stage,
            'snn': {
                'spiking': spiking,
                'beta': beta,
                'surrogate_alpha':  surrogate_alpha
            }
        }

        # Build the module
        self.spike_grad = surrogate.atan(alpha=surrogate_alpha) if spiking else None

        self.conv1 = nn.Conv2d(in_shape[0], initial_channels,
                               kernel_size=3, stride=initial_stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.act1 = new_activation(spiking, beta, self.spike_grad, threshold)

        self.stages = nn.ModuleDict()
        current_channels = initial_channels
        for s, out_channels in enumerate(stage_channels):
            num_blocks = num_blocks_per_stage
            if isinstance(num_blocks_per_stage, (list, tuple)):
                num_blocks = num_blocks_per_stage[s]
            cur_stride = stage_stride
            if isinstance(stage_stride, (list, tuple)):
                cur_stride = stage_stride[s]
            stage = new_stage(ResBlock, current_channels, out_channels, num_blocks,
                              cur_stride, padding, spiking, beta, self.spike_grad, threshold)
            self.stages.add_module(f'stage{s+1}', stage)
            current_channels = out_channels

        if self.neglect:
            for c in range(len(stage_channels)):
                pool = 4 + len(stage_channels) - c
                self.__setattr__(f'classifier{c+1}', Classifier(1, out_classes, pool=pool))
                self.init_flat_features(f'classifier{c+1}', in_shape, out_classes, module=Classifier,
                                        module_args={'pool': pool,
                                                     'cost': (c*neglect)/len(stage_channels)},)
        else:
            self.avg_pool = nn.AvgPool2d(4)
            self.linear = nn.Linear(1, out_classes)
            self.init_flat_features('linear', in_shape, out_classes)

    def forward(self, x, num_steps=5, stop_threshold=1., stop_after=None, include_first_classifier=False):
        if self.spiking:
            out = self._spike_forward(x, num_steps, stop_threshold=stop_threshold, stop_after=stop_after,
                                      include_first_classifier=include_first_classifier)
        else:
            out = self._forward(x, stop_threshold=stop_threshold, stop_after=stop_after,
                                include_first_classifier=include_first_classifier)
        return out

    def _spike_forward(self, x, num_steps=5, *args, **kwargs):
        self.spike_reset()
        spk_rec = []
        out = None
        for step in range(num_steps):
            out = self._forward(x, *args, **kwargs)
            spk_rec.append(out)
        spk_rec = torch.stack(spk_rec, dim=2)
        out = torch.mean(spk_rec, dim=-1)
        return out

    def _forward(self, x, stop_threshold=1., stop_after=None, include_first_classifier=False):
        out = self.quant(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)
        com_out = torch.zeros(x.shape[0], self.out_classes, device=self.device)
        c_count = 0
        for c, (stage_name, stage) in enumerate(self.stages.items()):
            out = stage(out)
            if (include_first_classifier or c > 0) and self.neglect:
                classifier = self.__getattr__(f'classifier{c+1}')
                cls_out, stopped = classifier(out, stop_threshold=stop_threshold)
                com_out += cls_out
                c_count += 1
                if stop_after is not None:
                    stopped = stop_after == c+1
                if not self.training and stopped:
                    com_out /= c_count
                    break
        if self.training:
            com_out /= len(self.stages)
        if not self.neglect:
            out = self.avg_pool(out)
            out = out.reshape(out.size(0), -1)
            out = self.linear(out)
            out = self.dequant(out)
            return out
        else:
            return com_out

    def spike_reset(self):
        utils.reset(self)
        for key, module in self.stages.named_modules():
            if isinstance(module, ResBlock):
                module.spike_reset()

    def classifier_reset(self):
        if self.neglect:
            for c, (stage_name, stage) in enumerate(self.stages.items()):
                classifier = self.__getattr__(f'classifier{c+1}')
                classifier.reset()

    def classifier_stats(self):
        stats = {}
        if self.neglect:
            for c, (stage_name, stage) in enumerate(self.stages.items()):
                classifier = self.__getattr__(f'classifier{c+1}')
                stats[f'classifier{c+1}'] = classifier.stopped
        return stats
    def init_flat_features(self, attr, in_shape, out_size, module=nn.Linear, module_args=None):
        if module_args is None:
            module_args = {}
        try:
            tmp = torch.rand(1, *in_shape)
            self(tmp, include_first_classifier=True)
        except RuntimeError as e:
            flat_size = int(re.search('\(\d+x(\d+) and .*\)', str(e)).groups()[0])
            self.__setattr__(attr, module(flat_size, out_size, **module_args))

    def fuse_model(self, qat=False, inplace=True):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if qat else torch.ao.quantization.fuse_modules

        # Fuse the first conv and bn in the main model
        fused = self
        if not inplace:
            fused = deepcopy(self)
        fused.eval()
        fused = fuse_modules(fused, [['conv1', 'bn1']], inplace=True)
        fused.fused = True

        for name, module in fused.named_modules():
            if isinstance(module, ResBlock):
                if self.spiking:
                    fuse_modules(module, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
                else:
                    fuse_modules(module, [['conv1', 'bn1', 'act1'], ['conv2', 'bn2']], inplace=True)
            if name.endswith('shortcut') and hasattr(module, 'conv') and hasattr(module, 'bn'):
                fuse_modules(module, [ ['conv', 'bn']], inplace=True)

        return self if inplace else fused

    def quantize(self, qat=True, inplace=True):
        self.quant_params = self.num_parameters()
        self.quant_sparsity = self.weight_sparsity()
        self.quant_avg_weight = self.weight_average()

        if qat:
            model_prepared = self.eval()
            tmp_input = torch.randn(4, *self.in_shape)
            model_prepared(tmp_input)
            mdl = torch.ao.quantization.convert(model_prepared, inplace=inplace)
            mdl.quantized = True
        else:
            model_prepared = self.prepare_quantization(qat=False, inplace=inplace)
            tmp_input = torch.randn(4, *self.in_shape)
            model_prepared(tmp_input)
            mdl = torch.ao.quantization.convert(model_prepared, inplace=True)
            mdl.quantized = True
        return mdl

    def prepare_quantization(self, qat=True, inplace=True):
        self.eval()
        self.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        fuse_model = self.fuse_model(inplace=inplace)
        if qat:
            torch.set_flush_denormal(True)
            mdl = torch.ao.quantization.prepare_qat(fuse_model.train(), inplace=True)
            tmp_input = torch.randn(4, *self.in_shape).to(self.device)
            mdl(tmp_input)
            mdl.quant_prepare = True
            mdl.quant_aware = True
        else:
            mdl = torch.ao.quantization.prepare(fuse_model, inplace=True)
            tmp_input = torch.randn(4, *self.in_shape).to(self.device)
            mdl(tmp_input)
            mdl.quant_prepare = True
        return mdl

    def prune(self, random=False, *args, **kwargs):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if random:
                    prune.random_unstructured(module, *args, **kwargs)
                else:
                    prune.l1_unstructured(module, *args, **kwargs)
            elif isinstance(module, torch.nn.Linear):
                if random:
                    prune.random_unstructured(module, *args, **kwargs)
                else:
                    prune.l1_unstructured(module, *args, **kwargs)

    def apply_pruning(self, inplace=True):
        mdl = self
        if not inplace:
            mdl = deepcopy(self)
        for name, module in mdl.named_modules():
            if torch.nn.utils.prune.is_pruned(module):
                if isinstance(module, nn.Conv2d):
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.remove(module, 'weight')
        mdl.pruned = False
        return mdl

    def num_parameters(self, non_zero=True):
        if self.quantized:
            return self.quant_params
        mdl = self
        if self.pruned:
            mdl = deepcopy(self)
            mdl.apply_pruning()
        total = 0
        for name, module in mdl.named_modules():
            for param_name, param in module.named_parameters():
                if non_zero:
                    total += torch.count_nonzero(param).item()
                else:
                    total += torch.numel(param)
        return total

    def model_size(self):
        """Returns the models size in Mb"""
        path = 'temp.p'
        torch.save(self.state_dict(), path)
        size = os.path.getsize(path) / 1e6
        os.remove(path)
        return size

    def weight_sparsity(self):
        if self.quantized:
            return self.quant_sparsity
        sparsity = 0
        num_modules = 0
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                weights = module.weight.data
                zero_weights = torch.sum(weights == 0)
                total_weights = weights.numel()
                sparsity += zero_weights.float() / total_weights
                num_modules += 1
        if num_modules == 0:
            num_modules = 1
        return float(sparsity / num_modules)

    def weight_average(self):
        if self.quantized:
            return self.quant_avg_weight
        significance = 0
        num_modules = 0
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                weights = module.weight.data
                num_non_zero = torch.sum(weights != 0)
                significance += torch.sum(torch.abs(weights)) / num_non_zero
                num_modules += 1
        if num_modules == 0:
            num_modules = 1
        return float(significance / num_modules)

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        # Load parameters into dict
        state_dict = super().state_dict(*args, **kwargs)

        # Save state
        state = dict()
        state['config'] = self.config
        state['hyperparameters'] = self.hyperparameters            # for additional info - not used for loading
        state['fused'] = self.fused,
        state['quant_prepare'] = self.quant_prepare,
        state['quant_aware'] = self.quant_aware,
        # state['quantized'] = self.quantized,
        state['pruned'] = self.pruned,
        state['neglect'] = self.neglect
        state['prune_arg'] = self.prune_arg,
        state['quant_sparsity'] = self.quant_sparsity,
        state['quant_params'] = self.quant_params,
        state['quant_avg_weight'] = self.quant_avg_weight,
        state['pruning_state'] = {name: module.state_dict() for name, module in self.named_modules()
                                       if isinstance(module, torch.nn.utils.prune.PruningContainer)},
        state_dict['state'] = state
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if 'model_state' in state_dict.keys():
            state_dict = state_dict['model_state']

        # Set state
        state = state_dict.pop('state', default={})
        do_fused = state.get('fused', (False,))[0]
        do_quant_prepare = state.get('quant_prepare', (False,))[0]
        do_quant_aware = state.get('quant_aware', (False,))[0]
        do_quantized = state.get('quantized', (False,))[0]
        do_pruned = state.get('pruned', (False,))[0]
        do_prune_arg = state.get('prune_arg', (([], {}),))[0]
        # do_neglect = state.get('neglect', 0.)
        self.quant_sparsity = state.get('quant_sparsity', 0.)
        self.quant_params = state.get('quant_params', 0)
        self.quant_avg_weight = state.get('quant_avg_weight', 0.)
        pruning_state = state.get('pruning_state', None)

        # for name, module in self.named_modules():
        #     if isinstance(module, torch.nn.utils.prune.PruningContainer) and name in pruning_state:
        #         module.load_state_dict(state_dict[name])

        # Load state
        if do_pruned:
            args, kwargs = do_prune_arg if isinstance(do_prune_arg, (list, tuple)) and len(do_prune_arg) == 2 else ([], {})
            if len(args) > 0:
                self.prune(*args, **kwargs)
        if do_quantized:
            self.quantize(qat=self.quant_aware, inplace=True)
        elif do_quant_aware:
            pass # self.prepare_quantization(qat=True, inplace=True)
        elif do_quant_prepare:
            self.prepare_quantization(qat=False, inplace=True)
        elif do_fused:
            self.fuse_model(qat=do_quant_aware, inplace=True)

        # For backwards compatibility
        new_dict = {}
        for key, value in state_dict.items():
            key = key.replace('shortcut.0', 'shortcut.conv')
            key = key.replace('shortcut.1', 'shortcut.bn')
            key = key.replace('leaky', 'act')
            key = key.replace('relu', 'relu')
            if self.neglect and key in ('linear', 'avg_pool'):
                continue
            new_dict[key] = value

        # Load parameters
        super().load_state_dict(new_dict, False)

    @staticmethod
    def from_state_dict(state_dict: dict, strict: bool = True, **kwargs):
        """Initializes the model with the correct parameters to load the state_dict."""

        if 'model_state' in state_dict.keys():
            state_dict = state_dict['model_state']
        state = state_dict.get('state', {})
        config = state.get('config', {'in_shape': [1, 128, 111], 'initial_channels': 128,
                                      'stage_channels': [128, 128, 256, 512],
                                      'stage_stride': [1, 2, 2, 2]})
        warnings.warn('Model configuration was empty! Using default configuration')
        config.update(kwargs)
        new_model = ResNet(**config)
        new_model.load_state_dict(state_dict, strict)
        return new_model

    def __deepcopy__(self, memodict: dict = None):
        copy_obj = self
        trials = 0
        while trials < 10:
            try:
                path = 'temp.p'
                torch.save(self.state_dict(), path)
                state_dict_copy = torch.load(path)
                copy_obj = ResNet(**self.config)
                copy_obj.load_state_dict(state_dict_copy)
                os.remove(path)
                break
            except FileNotFoundError as e:
                trials += 1
                continue
        return copy_obj


class TestResNetModel(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_flush_denormal(True)

        input_shape = (1, 128, 111)
        cls.sample = torch.rand(1, *input_shape)
        cls.states = torch.load('../models/resnet_ep55_acc99_sprs0.pt', map_location='cpu')

    def test_prediction(self):
        model = ResNet.from_state_dict(self.states)

        st = time.perf_counter()
        prediction = model(self.sample)
        et = time.perf_counter() - st

        self.assertIsNotNone(prediction, "Prediction should not be None")
        print('--- Initial Prediction ---')
        print(f'Took: {et:.5f} s')

    def test_model_pruning(self):
        model = ResNet.from_state_dict(self.states)

        model.prune(name='weight', amount=0.95)
        model.apply_pruning()

        st = time.perf_counter()
        prediction = model(self.sample)
        et = time.perf_counter() - st

        self.assertIsNotNone(prediction, "Prediction after pruning should not be None")
        print('\n--- After Pruning ---')
        print(f'Took: {et:.5f} s')

    def test_qat(self):
        model = ResNet.from_state_dict(self.states)
        if not model.spiking:
            # Save the size of the model before quantization
            original_size = model.model_size()

            # Prepare and perform quantization
            model_qat = model.prepare_quantization(qat=True, inplace=True)
            model_int8 = model_qat.quantize(inplace=True)

            # Save the size of the model after quantization
            quantized_size = model_int8.model_size()

            # Perform a prediction to ensure model functionality
            st = time.perf_counter()
            prediction = model_int8(self.sample)
            et = time.perf_counter() - st

            # Assert that the prediction is not None
            self.assertIsNotNone(prediction, "Prediction after quantization should not be None")

            # Assert that the quantized model is smaller
            self.assertLess(quantized_size, original_size, "Quantized model should be smaller than the original model")

            # Print the results
            print('\n--- After Quantization ---')
            print(f'Took: {et:.5f} s')
            print(f'Original Model Size: {original_size:.3f} Mb')
            print(f'Quantized Model Size: {quantized_size:.3f} Mb')

    def save_load_model(self, device, prune=False, qat=False, spiking=False):
        model = ResNet.from_state_dict(self.states)
        if qat:
            model = model.prepare_quantization(qat=True, inplace=True)
        if prune:
            model.prune('weight', amount=0.95)

        torch.save(model.state_dict(), f'../models/test_{str(device).lower()}.pt')
        model2 = ResNet.from_state_dict(torch.load(f'../models/test_{str(device).lower()}.pt', map_location=device))
        sps1, sps2 = model.weight_sparsity(), model2.weight_sparsity()
        sz1, sz2 = model.model_size(), model2.model_size()

        print(f'Sparsity: {sps1} {sps2}')
        print(f'Size: {sz1} {sz2}')
        self.assertAlmostEquals(sps1, sps2, places=4, msg=f"Model sparsity does not match")
        self.assertAlmostEquals(sz1, sz2, delta=1, msg=f"Model size does not match")
        return model2.to(device)

    def test_model_on_cpu(self):
        model_cpu = self.save_load_model('cpu')
        prediction_cpu = model_cpu(self.sample)

        model_cpu = self.save_load_model('cpu', prune=True)
        prediction_cpu = model_cpu(self.sample)

        model_cpu = self.save_load_model('cpu', qat=True)
        prediction_cpu = model_cpu(self.sample)

        model_cpu = self.save_load_model('cpu', prune=True, qat=True)
        prediction_cpu = model_cpu(self.sample)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU is not available")
    def test_model_on_gpu(self):
        model_gpu = self.save_load_model('cuda')
        gpu_sample = self.sample.to('cuda')
        prediction_gpu = model_gpu(gpu_sample)

        model_gpu = self.save_load_model('cuda', prune=True)
        prediction_gpu = model_gpu(gpu_sample)

        model_gpu = self.save_load_model('cuda', qat=True)
        prediction_gpu = model_gpu(gpu_sample)

        model_gpu = self.save_load_model('cuda', prune=True, qat=True)
        prediction_gpu = model_gpu(gpu_sample)


if __name__ == '__main__':
    torch.set_flush_denormal(True)      # Important: sets small tensor values to zero

    input_shape = (1, 128, 111)
    model = ResNet(input_shape,
                   initial_channels=128,
                   stage_channels=(128, 128, 256, 512),
                   stage_stride=(1, 2, 2, 2),
                   num_blocks_per_stage=2,
                   spiking=False)

    torch.save(model.state_dict(), '../models/test.pt')
    model = ResNet.from_state_dict(torch.load('../models/test.pt', map_location='cpu'))

    # model.load_state_dict(torch.load('../models/resnnet_ep20_acc88_sprs0.pt', map_location='cpu'))
    model.load_state_dict(torch.load('../models/resnet_ep55_acc99_sprs0.pt', map_location='cpu'))

    sample = torch.rand(1, *input_shape)

    st = time.perf_counter()
    prediction = model(sample)
    et = time.perf_counter() - st

    print('--- Run 1 ---')
    print(f'Took: {et:.5f} s')
    print(f'Num Params: {model.num_parameters()}')
    print(f'Sparsity: {model.weight_sparsity()*100:.3f} %')
    print(f'Size: {model.model_size():.3f} Mb')
    print(prediction)

    model.prune('weight', amount=0.95)
    model.apply_pruning()

    # torch.save(model.state_dict(), '../models/test.pt')
    # model.load_state_dict(torch.load('../models/test.pt', map_location='cpu'))

    st = time.perf_counter()
    prediction = model(sample)
    et = time.perf_counter() - st

    print('\n--- Run 2 ---')
    print(f'Took: {et:.5f} s')
    print(f'Num Params: {model.num_parameters()}')
    print(f'Sparsity: {model.weight_sparsity()*100:.3f} %')
    print(f'Size: {model.model_size():.3f} Mb')
    print(prediction)

    if not model.spiking:
        model_qat = model.prepare_quantization(qat=True, inplace=True)
        model_int8 = model_qat.quantize(inplace=True)

        st = time.perf_counter()
        prediction = model_int8(sample)
        et = time.perf_counter() - st

        print('\n--- Run 3 ---')
        print(f'Took: {et:.5f} s')
        print(f'Num Params: {model_int8.num_parameters()}')
        print(f'Sparsity: {model_int8.weight_sparsity()*100:.3f} %')
        print(f'Size: {model_int8.model_size():.3f} Mb')
        print(prediction)

