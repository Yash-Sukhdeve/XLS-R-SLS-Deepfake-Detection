"""Training and inference entry point for SLS + XLS-R deepfake detector.

Usage:
    # Training
    python -m sls_asvspoof.train --config configs/train_df.yaml

    # Evaluation (DF track)
    python -m sls_asvspoof.train --track=DF --is_eval --eval \\
        --model_path=models/epoch_2.pth \\
        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt \\
        --database_path=/path/to/ASVspoof2021_DF_eval/ \\
        --eval_output=scores/scores_DF.txt
"""

import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from sls_asvspoof.data_utils import (
    genSpoof_list, Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval, Dataset_in_the_wild_eval
)
from sls_asvspoof.model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms


def evaluate_accuracy(dev_loader, model, device, loss_weights):
    """Compute validation loss and accuracy.

    Args:
        dev_loader: DataLoader for validation set.
        model: The model to evaluate.
        device: Device string ('cuda' or 'cpu').
        loss_weights: List of [bonafide_weight, spoof_weight] for WCE.

    Returns:
        Tuple of (val_loss, val_accuracy_percent).
    """
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    return val_loss, acc


def produce_evaluation_file(dataset, model, device, save_path, eval_batch_size=4):
    """Generate score file for evaluation.

    Args:
        dataset: Evaluation dataset.
        model: Trained model.
        device: Device string.
        save_path: Path to save scores.
        eval_batch_size: Batch size for evaluation (default: 4).
    """
    data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    model.eval()

    for batch_x, utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, model, lr, optimizer, device, loss_weights):
    """Train for one epoch.

    Args:
        train_loader: DataLoader for training set.
        model: The model to train.
        lr: Learning rate (for logging; optimizer already configured).
        optimizer: The optimizer.
        device: Device string.
        loss_weights: List of [bonafide_weight, spoof_weight] for WCE.

    Returns:
        Average training loss for the epoch.
    """
    running_loss = 0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in tqdm(train_loader):

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)

        running_loss += (batch_loss.item() * batch_size)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total

    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLS + XLS-R Audio Deepfake Detection')
    # Dataset
    parser.add_argument('--database_path', type=str, default='./data/',
                        help='Base directory for audio data (contains ASVspoof2019_LA_train/, etc.)')
    parser.add_argument('--protocols_path', type=str, default='./database/',
                        help='Base directory for protocol files')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--loss_weights', type=float, nargs=2, default=[0.1, 0.9],
                        help='Class weights for WCE loss [bonafide, spoof]')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without val_loss improvement)')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for evaluation (reduce if OOM)')
    parser.add_argument('--audio_max_len', type=int, default=64600,
                        help='Max audio length in samples (~4 sec at 16kHz)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate in Hz')
    parser.add_argument('--xlsr_model', type=str, default='xlsr2_300m.pt',
                        help='Path to XLS-R pretrained checkpoint')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers')

    # Model
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint for evaluation or resume')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')

    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF',
                        choices=['LA', 'In-the-Wild', 'DF'], help='Evaluation track')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Run evaluation mode')
    parser.add_argument('--is_eval', action='store_true', default=False,
                        help='Use eval database')
    parser.add_argument('--eval_part', type=int, default=0)

    # Backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false',
                        default=True,
                        help='Use cudnn-deterministic? (default true)')
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true',
                        default=False,
                        help='Use cudnn-benchmark? (default false)')

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3,
                    help='Rawboost algos: 0=None, 1=LnL, 2=ISD, 3=SSI, 4=1+2+3, 5=1+2, 6=1+3, 7=2+3, 8=1||2')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                    help='Number of notch filters [default=5]')
    parser.add_argument('--minF', type=int, default=20,
                    help='Minimum centre frequency [Hz] of notch filter [default=20]')
    parser.add_argument('--maxF', type=int, default=8000,
                    help='Maximum centre frequency [Hz] (<sr/2) of notch filter [default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                    help='Minimum width [Hz] of filter [default=100]')
    parser.add_argument('--maxBW', type=int, default=1000,
                    help='Maximum width [Hz] of filter [default=1000]')
    parser.add_argument('--minCoeff', type=int, default=10,
                    help='Minimum filter coefficients [default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                    help='Maximum filter coefficients [default=100]')
    parser.add_argument('--minG', type=int, default=0,
                    help='Minimum gain factor of linear component [default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                    help='Maximum gain factor of linear component [default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                    help='Minimum gain difference between linear and non-linear components [default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                    help='Maximum gain difference between linear and non-linear components [default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                    help='Order of the (non-)linearity where N_f=1 refers only to linear components [default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                    help='Maximum number of uniformly distributed samples in [%%] [default=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                    help='Gain parameters > 0 [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                    help='Minimum SNR value for coloured additive noise [default=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                    help='Maximum SNR value for coloured additive noise [default=40]')

    ##===================================================Rawboost data augmentation ======================================================================#

    # YAML config support
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    # Load YAML config: only override args that are still at their default values
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        defaults = parser.parse_args([])
        for k, v in yaml_config.items():
            if hasattr(args, k) and getattr(args, k) == getattr(defaults, k):
                setattr(args, k, v)

    #make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model = nn.DataParallel(model).to(device)
    print('nb_params:', nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    # evaluation mode on the In-the-Wild dataset.
    if args.track == 'In-the-Wild':
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output,
                                eval_batch_size=args.eval_batch_size)
        sys.exit(0)

    # evaluation mode on the DF or LA dataset.
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output,
                                eval_batch_size=args.eval_batch_size)
        sys.exit(0)


    # define train dataloader
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof_DF_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False
    )

    print('no. of training trials', len(file_train))

    train_set = Dataset_ASVspoof2019_train(
        args, list_IDs=file_train, labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train'),
        algo=args.algo
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, drop_last=True)

    del train_set, d_label_trn


    # define dev (validation) dataloader
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'),
        is_train=False, is_eval=False
    )

    print('no. of validation trials', len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(
        args, list_IDs=file_dev, labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_dev'),
        algo=args.algo
    )

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)

    del dev_set, d_label_dev



    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):

        running_loss = train_epoch(train_loader, model, args.lr, optimizer, device, args.loss_weights)
        val_loss, val_acc = evaluate_accuracy(dev_loader, model, device, args.loss_weights)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1


        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - train_loss: {} - val_loss: {} - val_acc: {}'.format(epoch,
                                                   running_loss, val_loss, val_acc))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))

        if patience_counter >= args.patience:
            print("Early stopping triggered, best model is epoch: ", epoch - patience_counter)
            break
