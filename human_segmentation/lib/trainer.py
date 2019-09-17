"""Human segmentation training, inference and evaluation"""
import datetime
import json
import os

import pandas as pd  # type: ignore
import torch  # type: ignore
from tqdm import tqdm as tqdm  # type: ignore

from lib.dataset import *
from lib.model import *
from lib.loss import *
# type: ignore


class Trainer:
    """
    Main class

    Parameters
    ----------
    config : dict
        see configs/example.json for fields

    Returns
    -------
    None
    """
    def __init__(self, config):
        self.description = config['description']

        self.train_batch_size = config['train_batch_size']

        self.inference_batch_size = config['inference_batch_size']

        dataset_type = config['dataset_type']
        dataset = globals()[f'HumanSegmentationDataset{dataset_type}']
        train_set_path = config['train_set_path']
        self.train_set = dataset(
            config,
            train_set_path,
            transform_flag=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=8
        )

        self.train_only_mode = 'val_set_path' not in config
        if not self.train_only_mode:
            val_set_path = config['val_set_path']
            self.val_set = dataset(
                config,
                val_set_path,
                transform_flag=False
            )
            self.val_loader = torch.utils.data.DataLoader(
                dataset=self.val_set,
                shuffle=False,
                batch_size=self.inference_batch_size,
                num_workers=8
            )

        model_name = config['model_name']
        self.model = globals()[f'HumanSegmentationModel{model_name}'](config)

        loss_name = config['loss_name']
        self.criterion = globals()[f'HumanSegmentationLoss{loss_name}']()

        optimizer_name = config['optimizer_name']
        optimizer_kwargs = config['optimizer_kwargs']
        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.model.parameters(),
            **optimizer_kwargs
        )

        milestones = [int(ms) for ms in config['milestones'].split(',')]
        gamma = config['gamma']
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones,
            gamma
        )

        self.log_dir = config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(os.path.join(self.log_dir, 'weights')):
            os.mkdir(os.path.join(self.log_dir, 'weights'))

        self.n_epochs = config['n_epochs']

        self.save_each = config['save_each']

        self.test_weights_path = os.path.join(self.log_dir, 'weights/last.pth')
        if 'test_weights_path' in config:
            self.test_weights_path = config['test_weights_path']

        test_set_path = config['test_set_path']
        self.test_set = dataset(
            config,
            test_set_path,
            transform_flag=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=self.inference_batch_size,
            num_workers=8
        )

        self.test_save_path = config['test_save_path']

    def train(self):
        """Train model"""
        tic = None
        log = dict()

        # self.model = self.model.cuda()
        # self.criterion = self.criterion.cuda()

        # torch.backends.cudnn.benchmark = True

        for epoch in range(self.n_epochs):
            log[epoch] = dict()

            if epoch == 1:
                tic = datetime.datetime.now()

            print(self.description)
            print(f'Epoch {epoch + 1} of {self.n_epochs}')

            lr = self.optimizer.param_groups[0]['lr']
            print(f'LR: {lr:.6f}')
            log[epoch]['lr'] = lr

            train_loss, train_score = self.run_epoch()
            print(f'Train loss: {train_loss:.4f}')
            print(f'Train score: {train_score:.2%}')
            log[epoch]['train_loss'] = train_loss
            log[epoch]['train_score'] = train_score

            if not self.train_only_mode:
                val_score = self.evaluate(self.val_set, self.val_loader)
                print(f'Score: {val_score:.2%}')

            if (epoch + 1) % self.save_each == 0:
                state_dict = self.model.state_dict()

                torch.save(
                    state_dict,
                    os.path.join(
                        self.log_dir,
                        f'weights/{str(epoch + 1).zfill(4)}.pth'
                    )
                )

                torch.save(
                    state_dict,
                    os.path.join(
                        self.log_dir,
                        'weights/last.pth'
                    )
                )

                with open(os.path.join(self.log_dir, 'log.json'), 'w') as f:
                    json.dump(log, f)

            if epoch > 0:
                toc = datetime.datetime.now()
                eta = toc + (self.n_epochs - (1 + epoch)) * (toc - tic) / epoch
                print(f'ETA: {eta}')

            print()

            self.scheduler.step()

    def run_epoch(self):
        """Train one epoch"""
        self.model.train()

        running_score = 0.0
        running_loss = 0.0

        n_samples = 0
        n_batches = 0

        progress_bar = tqdm(total=len(self.train_loader), desc='Train')

        for inputs, masks in self.train_loader:
            cur_batch_size, *_ = inputs.size()

            # inputs = inputs.cuda()
            # masks = masks.cuda()

            self.optimizer.zero_grad()
            pred_masks = self.model.forward(inputs)

            print(type(masks), type(pred_masks))
            score = cur_batch_size * get_dice(masks, pred_masks)

            running_score += score

            loss = self.criterion(masks, pred_masks)

            running_loss += loss.item()

            n_batches += 1
            n_samples += cur_batch_size

            loss.backward()

            self.optimizer.step()

            progress_bar.update()

        progress_bar.close()

        score = running_score / n_samples
        loss = running_loss / n_batches

        return loss, score

    def test(self):
        """Test model"""
        self.model.load_state_dict(
            torch.load(
                self.test_weights_path,
                map_location='cpu'
            )
        )
        # self.model = self.model.cuda()

        test_score = self.evaluate(
            self.test_set,
            self.test_loader,
            save_path=self.test_save_path
        )
        print(f'Test score: {test_score:.2%}')

    def evaluate(self, dataset, loader, save_path=None):
        """Inference and evaluate"""
        self.model.eval()

        running_score = 0.0

        n_samples = 0

        mask_rles = list()

        progress_bar = tqdm(total=len(loader), desc='Inference')

        with torch.no_grad():
            for inputs, masks in loader:
                cur_batch_size, *_ = inputs.size()

                # inputs = inputs.cuda()

                masks_pred = self.model.forward(inputs)
                for mask_pred in masks_pred:
                    mask_rles.append(encode_rle(mask_pred))

                if all([isinstance(mask, np.ndarray) for mask in masks]):
                    score = cur_batch_size * get_dice(masks, masks_pred)
                else:
                    score = 0.0

                running_score += score

                n_samples += cur_batch_size

                progress_bar.update()

        progress_bar.close()

        score = running_score / n_samples

        df = pd.DataFrame(columns=['id', 'rle_mask'])
        df['id'] = pd.Series(dataset.id, index=range(len(dataset)))
        df['rle_mask'] = pd.Series(mask_rles, index=range(len(dataset)))

        if isinstance(save_path, str):
            df.to_csv(save_path, index=False)

        return score
