import os
import json
import json5
import wandb as wb
import torch
import platform
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import copy
import time
import statistics

from src.resnet import ResNet
from src.dataset import GoogleSpeechCommandsDataset
from src.utils import get_device

def init_train_data(directory, cache, encoding, augment, data_sample_limit, batch_size, multicache=False):
    train_data = GoogleSpeechCommandsDataset(directory, cache, encoder=encoding, sample_limit=data_sample_limit,
                                             augment=augment, train=True)
    if multicache:
        train_data.precache_multi()
    else:
        train_data.precache()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_data, train_loader

def init_test_data(directory, cache, encoding, augment, data_sample_limit, batch_size, multicache=False):
    test_data = GoogleSpeechCommandsDataset(directory, cache, encoder=encoding, sample_limit=data_sample_limit,
                                            augment=augment, train=False)
    if multicache:
        test_data.precache_multi()
    else:
        test_data.precache()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_data, test_loader

def train(model, train_loader, optimizer, criterion, epoch,
          log_interval=10, num_steps=5, neglect_threshold=0.75, device='cpu'):
    print('--- Training ---')
    model.train()
    correct = 0
    for batch_idx, (data, target, target_idx, target_lbl, data_idx, pitch_shift) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, num_steps=num_steps, stop_threshold=neglect_threshold)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            current = batch_idx * len(data)
            total = len(train_loader.dataset)
            percent = 100. * batch_idx / len(train_loader)
            print(f'\tEpoch {epoch} [{current}/{total} ({percent:.0f}%)]\tLoss: {loss.item():.6f}')

            # Log in Weights & Biases
            epoch_progress = (epoch-1) + batch_idx/len(train_loader)
            wb.log({
                "Epoch": epoch_progress,
                "Batch Index": batch_idx,
                "Train Loss": loss.item()
            })

def distill(teacher, student, train_loader, optimizer, criterion, epoch,
            T=2, soft_target_loss_weight=0.5, ce_loss_weight=0.75,
            log_interval=10, num_steps=5, neglect_threshold=0.75, device='cpu'):
    teacher.eval()
    student.train()

    running_loss = 0.0
    for batch_idx, (data, target, target_idx, target_lbl, data_idx, pitch_shift) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Teacher model
        with torch.no_grad():
            teacher_logits = teacher(data, num_steps=num_steps, stop_threshold=neglect_threshold)

        # Student model
        student_logits = student(data, num_steps=num_steps, stop_threshold=neglect_threshold)

        # Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

        # Calculate the true label loss
        label_loss = criterion(student_logits, target)

        # Weighted sum of the two losses
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            current = batch_idx * len(data)
            total = len(train_loader.dataset)
            percent = 100. * batch_idx / len(train_loader)
            print(f'Epoch {epoch} [{current}/{total} ({percent:.0f}%)]\tLoss: {loss.item():.6f}')

            # Log in Weights & Biases
            epoch_progress = (epoch - 1) + batch_idx / len(train_loader)
            wb.log({
                "Epoch": epoch_progress,
                "Batch Index": batch_idx,
                "Train Loss": loss.item()
            })

def test(model, test_loader, criterion, epoch=None,
         top_acc=0., cpu_tests=0, cpu_time_limit=120, return_time=False,
         log_interval=None, num_steps=5, neglect_threshold=0.75, stop_after=None, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0

    cpu_time = 0
    cpu_time_mean = 0
    cpu_time_std = 0
    device_time = 0
    print('\n--- Test Results ---')
    quant_size = None
    with torch.no_grad():
        st = time.perf_counter()
        print(f'-> Accuracy Validation on {str(device).upper()}')
        for batch_idx, (data, target, target_idx, target_lbl, data_idx, pitch_shift) in enumerate(test_loader):
            data, target, target_idx = data.to(device), target.to(device), target_idx.to(device)
            output = model(data, num_steps=num_steps, stop_threshold=neglect_threshold, stop_after=stop_after)
            test_loss += criterion(output, target).item()

            # Get ACC
            pred = output.argmax(dim=1)
            correct += (pred == target_idx).sum().item()

            if log_interval is not None and batch_idx % log_interval == 0:
                current = batch_idx * len(data)
                total = len(test_loader.dataset)
                percent = 100. * batch_idx / len(test_loader)
                print(f'Test [{current}/{total} ({percent:.0f}%)]\tCorrect: {correct}/{total}\tACC: {(correct / total)*100:.2f}%\tEpoch: {epoch}')

        device_time = (time.perf_counter() - st) / (test_loader.batch_size * len(test_loader))

        if not model.quant_aware and cpu_tests > 0:
            print('-> Performance Testing on CPU')
            cpu_model = copy.deepcopy(model)
            cpu_model.to("cpu")
            cpu_model.eval()
            # cpu_model.apply_pruning()

            # Cache the model
            print('\t-> Model caching...')
            num_samples = cpu_tests
            if cpu_tests > test_loader.batch_size:
                # cpu_tests = test_loader.batch_size
                num_samples = 1
            sample = None
            for c, (data, target, target_idx, target_lbl, data_idx, pitch_shift) in enumerate(test_loader):
                if sample is not None:
                    sample = torch.cat([sample, data[0:num_samples+1]], dim=0)
                else:
                    sample = data[0:num_samples+1]
                if sample.shape[0] > cpu_tests:
                    break
            output = cpu_model(sample[0:1], num_steps=num_steps)

            # Now count the time
            print('\t-> CPU time testing...')
            st = run_st = time.perf_counter()
            c = 0
            cpu_times = []
            cpu_time = 0
            while c < cpu_tests and (c == 0 or cpu_time < cpu_time_limit):
                run_st = time.perf_counter()
                output = cpu_model(sample[c:c+1], num_steps=num_steps,
                                   stop_threshold=neglect_threshold, stop_after=stop_after)
                run_el = time.perf_counter() - run_st
                c += 1
                cpu_time += run_el
                cpu_times.append(run_el)
                print(f'\t\t{c}. run ({run_el:.5}s)')
            tensor = torch.tensor(data, dtype=torch.float32)
            cpu_time_mean = cpu_time / c
            if len(cpu_times) > 1:
                cpu_time_std = statistics.stdev(cpu_times)

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    sparsity = model.weight_sparsity()
    avg_weight = model.weight_average()
    num_params = model.num_parameters()
    model_size = model.model_size()

    print(f'\n\tAvg. Loss:\t\t{test_loss:.6f}')
    print(f'\tAccuracy:\t\t{accuracy*100:.2f} % ({correct}/{len(test_loader.dataset)})')
    print(f'\tSparsity:\t\t{sparsity*100:.2f} %')
    print(f'\tAvg. Weight:\t{avg_weight:.2f}')
    print(f'\tNum. Params:\t{num_params}')
    quant_str = f'-> Quantized: {quant_size} ({(quant_size/model_size)*100:.2f} %)' if quant_size is not None else ''
    print(f'\tModel Size:\t\t{model_size:.2f} Mb {quant_str}')
    print(f'\tDevice Time:\t{device_time:6f} s')
    if cpu_tests > 0:
        print(f'\tCPU Time:\t\t{cpu_time_mean:.4f}±{cpu_time_std:.2f} s')
    print()

    if epoch is not None:
        distr = {}
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                class_name = layer.__class__.__name__
                weights = layer.weight.data.cpu().numpy()
                non_zero_weights = weights[weights != 0]
                distr[f'{name} ({class_name})'] = wb.Histogram(non_zero_weights.flatten(), num_bins=65)
        wb.log({
            "Epoch": epoch,
            "Test Accuracy": accuracy,
            "Test Loss": test_loss,
            "Sparsity": sparsity,
            "Average Weight": avg_weight,
            "Model Size (Mb)": model_size,
            "Number of Parameters": num_params,
            "Device Time": device_time,
            "CPU Time": cpu_time_mean,
            "CPU Time STD": cpu_time_std,
            "Top Test Accuracy": accuracy if accuracy > top_acc else top_acc,
            **distr
        })

    if return_time:
        return accuracy, sparsity, avg_weight, device_time, cpu_time_mean, cpu_time_std, num_params
    return accuracy, sparsity, avg_weight

def save_checkpoint(model: ResNet, optimizer, epoch, loss=None, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filepath, model=ResNet, optimizer=None, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)

    mdl_state = checkpoint.get('model_state')
    opt_state = checkpoint.get('optimizer_state')
    if mdl_state is None:
        # Treat as raw model file
        mdl_state = checkpoint
    if isinstance(model, ResNet):
        model.load_state_dict(mdl_state).to(device)
    else:
        model = ResNet.from_state_dict(mdl_state).to(device)

    if opt_state is not None:
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        optimizer.load_state_dict(opt_state)

    epoch = checkpoint.get('epoch', 1)
    loss = checkpoint.get('loss', 0)
    return epoch, loss, model, optimizer

def load_config(configuration, run_name_index=None):
    model_type = configuration['model']['type']
    model_initial_channels = configuration['model']['initial_channels']
    model_stage_channels = configuration['model']['stage_channels']
    model_stage_stride = configuration['model']['stage_stride']
    model_num_blocks_per_stage = configuration['model']['num_blocks_per_stage']
    model_padding = configuration['model']['padding']
    target_device = configuration['device']['target']

    data_train_dir = configuration['data']['train_dir']
    data_test_dir = configuration['data']['test_dir']
    data_cache_dir = configuration['data']['cache_dir']
    if not os.path.exists(data_train_dir.replace('*', '')):
        ab = data_train_dir.split('*/')
        if len(ab) == 2:
            data_train_dir = os.path.join('./', ab[1])
    if not os.path.exists(data_test_dir.replace('*', '')):
        ab = data_test_dir.split('*/')
        if len(ab) == 2:
            data_test_dir = os.path.join('./', ab[1])
    if not os.path.exists(data_cache_dir.replace('*', '')):
        ab = data_cache_dir.split('*/')
        if len(ab) == 2:
            data_cache_dir = os.path.join('./', ab[1])
    data_train_dir = data_train_dir.replace('*', '')
    data_test_dir = data_test_dir.replace('*', '')
    data_cache_dir = data_cache_dir.replace('*', '')

    data_encoding = configuration['data']['encoding']
    data_augment = configuration['data']['augment']
    data_sample_limit = configuration['data']['sample_limit']
    use_checkpoint = configuration['checkpoint']['use_checkpoint']
    checkpoint_path = configuration['checkpoint']['checkpoint_path']
    optim_type = configuration['training']['optimizer']
    learning_rate = configuration['training']['learning_rate']
    weight_decay = configuration['training']['weight_decay']
    batch_size = configuration['training']['batch_size']
    epochs = configuration['training']['epochs']
    prune = configuration['pruning']['prune']
    prune_interval = configuration['pruning']['prune_interval']
    prune_amount = configuration['pruning']['prune_amount']
    do_spike = configuration['spiking']['spike']
    spike_threshold = configuration['spiking']['threshold']
    spike_beta = configuration['spiking']['beta']
    spike_surrogate_alpha = configuration['spiking']['surrogate_alpha']
    spike_steps = configuration['spiking']['steps']
    quantize = configuration['quantization']['quantize']
    quantization_aware = configuration['quantization']['quantization_aware']
    do_neglect = configuration['neglecting']['neglect']
    neglect = configuration['neglecting']['amount']
    neglect_thresh = configuration['neglecting']['threshold']
    do_distillation = configuration['distillation']['distill']
    distill_teacher_path = configuration['distillation']['teacher_path']
    distill_t = configuration['distillation']['t']
    distill_soft_target_loss_weight = configuration['distillation']['soft_target_loss_weight']
    distill_ce_loss_weight = configuration['distillation']['ce_loss_weight']
    log_interval = configuration['logging']['log_interval']
    log_project = configuration['logging']['project']
    log_name = configuration['logging']['name']
    log_comment = configuration['logging']['comment']
    log_group = configuration['logging']['group']
    if run_name_index is not None:
        log_name += f' {run_name_index}'
        configuration['logging']['name'] = log_name
    checkpoints = configuration['logging']['checkpoints']
    checkpoint_interval = configuration['logging']['checkpoint_interval']
    if not do_neglect:
        neglect = 0
    return (model_type, model_initial_channels, model_stage_channels, model_stage_stride, model_num_blocks_per_stage,
            model_padding, target_device, data_train_dir, data_test_dir, data_cache_dir, data_encoding, data_augment,
            data_sample_limit, use_checkpoint, checkpoint_path, optim_type, learning_rate, weight_decay, batch_size,
            epochs, prune, prune_interval, prune_amount, do_spike, spike_steps, spike_threshold, spike_beta,
            spike_surrogate_alpha, quantize, quantization_aware, do_neglect, neglect, neglect_thresh, do_distillation,
            distill_teacher_path, distill_t, distill_soft_target_loss_weight, distill_ce_loss_weight, log_interval,
            log_project, log_name, log_comment, log_group, checkpoints, checkpoint_interval)

def execute(config_directory: str, root='./', run_name_index=None, multicache=False):
    torch.set_flush_denormal(True)      # Important: sets small tensor values to zero
    ct = datetime.now()

    # Load config
    with open(config_directory, 'r') as f:
        config = json5.load(f)
    (model_type, model_initial_channels, model_stage_channels, model_stage_stride, model_num_blocks_per_stage,
     model_padding, target_device, data_train_dir, data_test_dir, data_cache_dir, data_encoding, data_augment,
     data_sample_limit, use_checkpoint, checkpoint_path, optim_type, learning_rate, weight_decay, batch_size,
     epochs, prune, prune_interval, prune_amount, do_spike, spike_steps, spike_threshold, spike_beta,
     spike_surrogate_alpha, quantize, quantization_aware, do_neglect, neglect, neglect_thresh, do_distillation,
     distill_teacher_path, distill_t, distill_soft_target_loss_weight, distill_ce_loss_weight, log_interval,
     log_project, log_name, log_comment, log_group, checkpoints, checkpoint_interval) \
        = load_config(config, run_name_index=run_name_index)

    device, device_name = get_device(target=target_device)
    config['device']['name'] = device_name
    config['training']['start_time'] = ct.strftime("%Y-%m-%d_%H-%M-%S")

    # Dataset and DataLoader
    train_data, train_loader = init_train_data(data_train_dir, data_cache_dir, data_encoding,
                                               data_augment, data_sample_limit, batch_size, multicache)
    test_data, test_loader = init_train_data(data_test_dir, data_cache_dir, data_encoding,
                                             data_augment, data_sample_limit, batch_size, multicache)

    # Initialize model
    if use_checkpoint:
        # model = ResNet.from_state_dict(torch.load(checkpoint_path, map_location=device))
        epoch, loss, model, optimizer = load_checkpoint(checkpoint_path, device=device)
    else:
        model = ResNet(tuple(train_data.size()), len(train_data.labels),
                       initial_channels=model_initial_channels, stage_channels=model_stage_channels,
                       num_blocks_per_stage=model_num_blocks_per_stage, stage_stride=model_stage_stride,
                       padding=model_padding, spiking=do_spike, beta=spike_beta,
                       surrogate_alpha=spike_surrogate_alpha, threshold=spike_threshold,
                       neglect=neglect if do_neglect else 0).to(device)
    if optim_type == 'Checkpoint':
        pass
    elif optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer "{optim_type}" is not implemented!')
    criterion = nn.CrossEntropyLoss()

    # Initialize teacher
    teacher_model = None
    config['model']['hyperparameters'] = model.hyperparameters
    config['model']['config'] = model.config
    if do_distillation:
        teacher_model = ResNet.from_state_dict(torch.load(distill_teacher_path, map_location=device)).to(device)

    # Make model quantization aware
    if quantize and quantization_aware:
        model = model.prepare_quantization(qat=True, inplace=True)

    # Configure run directory
    config['logging']['config_dir'] = config_directory
    model_name = model_type.lower().replace('resnet', 'resnnet' if do_spike else "resnet")
    run_dir = config['logging']['run_dir'] = os.path.join(root, f'runs/{log_name.lower().replace(" ", "-")}'
                                                                f'_gr-{log_group.lower().replace(" ", "-")}_'
                                                                f'{model_name}_{ct.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(run_dir, exist_ok=True)
    if checkpoints:
        os.makedirs(os.path.join(run_dir, 'checkpoint'))
    with open(os.path.join(run_dir, f'run.json'), 'w') as f:
        json.dump(config, f, indent=3)
    print(f'--- CONFIG ({config_directory}) ---')
    print(json.dumps(config, indent=3))
    print()

    # Model memory
    if torch.cuda.is_available():
        print(f'Model uses {torch.cuda.memory_allocated(device) * 1e-6} Mb on GPU')

    # Init weights and biases
    wb.login()
    run = wb.init(dir=root, project=log_project, name=log_name, group=log_group, config=config, notes=log_comment)
    run.log_code()

    # Training loop
    top_acc = 0.
    top_check = ''
    accuracy, sparsity, significance = test(model, test_loader, criterion, 0, cpu_tests=16, device=device)
    best_accuracy = 0.
    for epoch in range(1, epochs+1):
        if do_distillation:
            # Knowledge Distillation
            distill(teacher_model, model, train_loader, optimizer, criterion,
                    epoch, T=distill_t, soft_target_loss_weight=distill_soft_target_loss_weight,
                    ce_loss_weight=distill_ce_loss_weight, log_interval=log_interval,
                    num_steps=spike_steps, neglect_threshold=neglect_thresh, device=device)
        else:
            # Training
            train(model, train_loader, optimizer, criterion, epoch, log_interval=log_interval,
                  num_steps=spike_steps, neglect_threshold=neglect_thresh, device=device)

        # Prune in regular intervals
        if prune and (epoch % prune_interval == 0) and epoch < epochs:
            model.prune(name='weight', amount=prune_amount)

        # Testing
        accuracy, sparsity, significance = test(model, test_loader, criterion, epoch, best_accuracy,
                                                num_steps=spike_steps, neglect_threshold=neglect_thresh,
                                                cpu_tests=16, device=device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if checkpoints and (accuracy > top_acc or (epoch-1) % checkpoint_interval == 0):
            path = os.path.join(run_dir, 'checkpoint',
                                f"{model_type.lower()}_ep{epoch}_acc{accuracy*100:.0f}_sprs{sparsity*100:.0f}"
                                f"{'_qat' if quantize and quantization_aware else ''}.pt")
            if len(top_check) > 0:
                if os.path.isfile(path):
                    os.remove(path)
            if accuracy > top_acc and (epoch-1) % checkpoint_interval != 0:
                top_check = path
            top_acc = accuracy
            # ap_model = model.apply_pruning(inplace=False)
            # torch.save(ap_model.state_dict(), path)
            save_checkpoint(model, optimizer, epoch, filename=path)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(run_dir, f'{model_type.lower()}_final.pt'))


if __name__ == '__main__':
    config_dir = '../config/baseline.json'

    execute(config_dir, root='../')
