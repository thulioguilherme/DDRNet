import datetime

import math

from pathlib import Path

import sys

import torch

from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from tqdm import tqdm

import yaml

# #{ include this project packages

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# #}

from models.ddrnets23slim import DDRNetS23slim, load_weights
from utils.datasets.cityscapes import Cityscapes
from utils.losses.ohem_loss import OhemCrossEntropyLoss
from utils.model_evaluator import ModelEvaluator
from utils.poly_lr_scheduler import PolyLRScheduler


# #{ read_config_file()

def read_config_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            return yaml_content
    except FileNotFoundError:
        print(f'Error: The file {file_path} was not found.')
        return None
    except yaml.YAMLError as exc:
        print(f'Error parsing YAML file: {exc}')
        return None

# #}


# #{ generate_model_filename()

def generate_model_filename(model_name, datetime_str, mean_iou):
    formatted_accuracy = f'{mean_iou:.1f}'.replace('.', '_')

    filename = f'{model_name}_mIoU_{formatted_accuracy}_{datetime_str}.pth'

    return filename

# #}


# #{ get_datetime_as_string()

def get_datetime_as_string():
    current_datetime = datetime.datetime.now()

    datetime_str = current_datetime.strftime('%Y%m%d_%H%M%S')

    return datetime_str

# #}


if __name__ == '__main__':

    # cleanup any residual memory from previous run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # allow CuDNN to auto-tune
    torch.backends.cudnn.benchmark = True

    scaler = GradScaler()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} as device')

    this_file_dir = Path(__file__).resolve().parent
    configs = read_config_file(this_file_dir / '../configs/cityscapes.yaml')['train']

    home_path = Path.home()
    root_path = home_path / '../app/data/cityscapes'

    datetime_str = get_datetime_as_string()
    log_dir = 'ddrnets_' + datetime_str
    writer = SummaryWriter(home_path / '../app/runs' / log_dir)
    global_step = 0

    # #{ prepare Cityspaces dataset loaders

    train_dataset = Cityscapes(root=root_path, partition='train')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=math.ceil(configs['batch_size'] / configs['accumulation_steps']),
        shuffle=True,
        num_workers=8,
        drop_last=configs['drop_last']
    )

    train_num_images = len(train_dataset)
    print(f'Number of images in train dataset: {train_num_images}')

    val_dataset = Cityscapes(root=root_path, partition='val')

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=8,
        drop_last=configs['drop_last']
    )

    val_num_images = len(val_dataset)
    print(f'Number of images in validation dataset: {val_num_images}')

    # #}

    ddrnets23slim = DDRNetS23slim(num_classes=19)

    criterion = OhemCrossEntropyLoss()

    optimizer = None
    if configs['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(
            ddrnets23slim.parameters(),
            lr=configs['optimizer']['learning_rate'],
            momentum=configs['optimizer']['momentum'],
            weight_decay=configs['optimizer']['weight_decay'],
            nesterov=True
        )
    else:
        print('Error: no valid optimizer')
        exit(1)

    steps_per_weight_update = 0
    true_batches = len(train_loader)
    if configs['drop_last']:
        steps_per_weight_update = true_batches // configs['accumulation_steps']
    else:
        steps_per_weight_update = math.ceil(true_batches / configs['accumulation_steps'])

    total_steps = steps_per_weight_update * configs['num_epochs']

    scheduler = PolyLRScheduler(
        optimizer,
        max_iters=total_steps,
        power=0.9
    )

    ddrnets23slim = ddrnets23slim.to(device)

    num_epochs = configs['num_epochs']
    for epoch in range(num_epochs):

        # #{ training loop

        ddrnets23slim.train()
        epoch_total_loss_sum = 0.0

        train_loader_tqdm = tqdm(
            train_loader,
            desc=f'(Train) Epoch {epoch+1}/{num_epochs}',
            unit='batch'
        )

        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = ddrnets23slim(inputs)
                    normal_loss = criterion(outputs[0], labels)
                    aux_loss = criterion(outputs[1], labels)
                    mini_batch_loss = normal_loss + 0.4 * aux_loss

                    if configs['accumulation_steps'] > 1:
                        loss_for_backward = mini_batch_loss / configs['accumulation_steps']
                    else:
                        loss_for_backward = mini_batch_loss

                    epoch_total_loss_sum += mini_batch_loss.item() * inputs.size(0)

                    scaler.scale(loss_for_backward).backward()

                    if (i + 1) % configs['accumulation_steps'] == 0:
                        scaler.step(optimizer)
                        scaler.update()

                        optimizer.zero_grad()

                        scheduler.step()

                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('Learning rate', current_lr, global_step)
                        global_step += 1

        if len(train_loader) % configs['accumulation_steps'] != 0:
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            scheduler.step()
            final_epoch_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning rate', final_epoch_lr, global_step)
            global_step += 1

        train_epoch_loss = epoch_total_loss_sum / len(train_loader.dataset)
        print('  Loss: {:.4f}'.format(train_epoch_loss))

        writer.add_scalar('Loss/train', train_epoch_loss, epoch + 1)

        # #}

        # #{ evaluation loop

        ddrnets23slim.eval()
        val_loss = 0.0

        val_loader_tqdm = tqdm(
            val_loader,
            desc=f'(Val) Epoch {epoch+1}/{num_epochs}',
            unit='batch'
        )

        with torch.no_grad():
            with autocast():
                for inputs, labels in val_loader_tqdm:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = ddrnets23slim(inputs)
                    normal_loss = criterion(outputs[0], labels)
                    aux_loss = criterion(outputs[1], labels)
                    loss = normal_loss + 0.4 * aux_loss

                    val_loss += loss.item() * inputs.size(0)

        val_loss_epoch = val_loss / len(val_loader.dataset)
        print('  Loss: {:.4f}'.format(val_loss_epoch))

        writer.add_scalar('Loss/val', val_loss_epoch, epoch + 1)

        # #}

        # cleanup any residual memory after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('Training finished!')

    model_evaluator = ModelEvaluator(num_classes=19, device=device)
    res = model_evaluator(ddrnets23slim, val_loader)
    mean_iou = res['mean_iou'] * 100
    print(f'Mean IoU: {mean_iou:.2f}%')

    model_filename = generate_model_filename('ddrnets23slim', datetime_str, mean_iou)
    torch.save(ddrnets23slim.state_dict(), this_file_dir / model_filename)
    print(f'Weights saved to {model_filename}!')

    writer.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.exit(0)
