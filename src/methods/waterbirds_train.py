from src.methods.base_method import TrainBaseMethod, Mode
from src.datasets import WaterbirdsDataset
from src.utils import load_checkpoint
from src.utils import AverageMeter, calculate_accuracy
from src.utils import change_column_value_of_existing_row
from tqdm import tqdm

import torchvision.transforms as transforms
import torch.nn.functional as F

import torch
import os
import pandas as pd

from src.similarity_based_explanation.metrics import cossim, l2sim, dotprod


class WaterbirdsTrain(TrainBaseMethod):
    def __init__(self, args) -> None:
        super().__init__(args)

    def prepare_data_loaders(self) -> None:
        scale = 256.0/224.0
        target_resolution = (224, 224)
        self.transform_test = transforms.Compose([
                transforms.Resize(
                    (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                    0.229, 0.224, 0.225])
        ])
        self.transform_data_to_mask = transforms.Compose([
            transforms.Resize(
                (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                    0.229, 0.224, 0.225])
        ])
        self.train_dataset = WaterbirdsDataset(raw_data_path=self.args.dataset_dir, root=os.path.join(
            self.args.base_dir, 'datasets', 'Waterbirds'), split='train', transform=self.transform_train, return_places=True)
        # shuffle = True
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.train_batch, shuffle=False, num_workers=self.args.workers)
        self.val_dataset = WaterbirdsDataset(raw_data_path=self.args.dataset_dir, root=os.path.join(
            self.args.base_dir, 'datasets', 'Waterbirds'), split='val', transform=self.transform_test, return_places=True)
        self.data_to_mask_dataset = WaterbirdsDataset(raw_data_path=self.args.dataset_dir, root=os.path.join(
            self.args.base_dir, 'datasets', 'Waterbirds'), split='train', transform=self.transform_data_to_mask)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.workers)
        self.data_to_mask_loader = torch.utils.data.DataLoader(
            self.data_to_mask_dataset, batch_size=self.args.masking_batch_size, shuffle=True, num_workers=self.args.workers)
        self.test_dataset = WaterbirdsDataset(raw_data_path=self.args.dataset_dir, root=os.path.join(
            self.args.base_dir, 'datasets', 'Waterbirds'), split='test', transform=self.transform_test, return_places=True)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.workers)

    def run_an_epoch_with_group(self, data_loader, epoch, mode: Mode = Mode.train):
        if mode == Mode.train:
            self.model.train()
        else:
            self.model.eval()

        ###
        losses = AverageMeter()
        accuracies = AverageMeter()
        all_predictions = []
        all_aux_labels = []
        all_labels = []
        
        with torch.set_grad_enabled(mode == Mode.train):
            progress_bar = tqdm(data_loader)
            self.logger.info(
                f"{mode.name} epoch: {epoch}"
            )
            for data in progress_bar:
                progress_bar.set_description(f'{mode.name} epoch {epoch}')
                inputs, labels, aux_labels = data[0], data[2], data[-1]
                inputs, labels = inputs.to(
                    self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                losses.update(loss.item(), inputs.size(0))
                output_probabilities = F.softmax(outputs, dim=1)
                probabilities, predictions = output_probabilities.data.max(1)
                accuracies.update(calculate_accuracy(labels, predictions), 1)
                all_predictions.append(predictions.detach().cpu())
                all_aux_labels.append(aux_labels)
                all_labels.append(labels.detach().cpu())
                if mode == Mode.train:
                    self.optimize(loss=loss)
                progress_bar.set_postfix(
                    {
                        "loss": losses.avg,
                        "accuracy": accuracies.avg,
                    }
                )
        all_predictions = torch.cat(all_predictions)
        all_aux_labels = torch.cat(all_aux_labels)
        all_labels = torch.cat(all_labels)
        ###
            
        train_labels = []
        train_aux_labels = []
        for i, data in enumerate(self.train_loader):
            labels, aux_labels = data[2], data[-1]
            train_labels.append(labels)
            train_aux_labels.append(aux_labels)
        train_labels = torch.cat(train_labels)
        train_aux_labels = torch.cat(train_aux_labels)

        if self.args.print_detail:
            d = {}
            d['labels'] = all_labels.cpu()
            d['predictions'] = all_predictions.cpu()
            d['aux_labels'] = all_aux_labels.cpu()

            # similarity
            # similarity_l2sim = self.get_relevance_by_hsim(data_loader, sim_func=l2sim, if_grad=True)
            # similarity_cossim = self.get_relevance_by_hsim(data_loader, sim_func=cossim, if_grad=True)
            # similarity_dotprod = self.get_relevance_by_hsim(data_loader, sim_func=dotprod, if_grad=True)

            # for name, similarity in zip(["dotprod", "cossim", "l2sim"], [similarity_dotprod, similarity_cossim, similarity_l2sim]):
            #     assert similarity.shape[0] == len(all_labels)
            #     train_rank_top = similarity.argmax(dim=1, keepdim=False)
            #     train_rank_top_label = [train_labels[i].item() for i in train_rank_top]
            #     train_rank_top_aux_label = [train_aux_labels[i].item() for i in train_rank_top]

            #     d['train_rank_top_' + name] = train_rank_top.cpu()
            #     d['train_rank_top_label_' + name] = train_rank_top_label
            #     d['train_rank_top_aux_label_' + name] = train_rank_top_aux_label
            
            # feature norm
            grad_norm, loss = self.feat_norm(self.test_loader, if_grad=True, flatten=False)
            feat_norm = self.feat_norm(self.test_loader, if_grad=False, flatten=False)

            # d['labels'] = train_labels.cpu()
            # d['aux_labels'] = train_aux_labels.cpu()

            d['grad_norm'] = grad_norm.cpu()
            d['feat_norm'] = feat_norm.cpu()
            d['loss'] = loss

            df = pd.DataFrame(d)
            df.to_csv('output_test_grad_loss_feat.csv', index=True)
            

        # groups = {
        #     0: [],
        #     1: [],
        #     2: [],
        #     3: [],
        # }
        # for aux_label, label, prediction in zip(all_aux_labels, all_labels, all_predictions):
        #     groups[2*aux_label.item()+label.item()].append(label.item()
        #                                                    == prediction.item())
        # weighted_acc = 0
        # accuracies = []
        # for group_id, group_predictions in groups.items():
        #     if len(group_predictions) == 0:
        #         accuracy = 0
        #     else: 
        #         accuracy = sum(group_predictions)/len(group_predictions)
        #     accuracies.append(accuracy)
        #     self.logger.info(f"accuracy of group {group_id+1} ({len(group_predictions)}): {accuracy}", print_msg=True)
        #     weighted_acc += accuracy*len(group_predictions)
        # weighted_acc /= len(all_predictions)
        # self.logger.info(f"average accuracy: {weighted_acc}", print_msg=True)
        return 0 #min(accuracies)

    def test(self, checkpoint_path=None):
        assert checkpoint_path is not None, "checkpoint path should be passed to the test function to test on that"
        self.logger.info("-" * 10 + "testing the model" +
                         "-" * 10, print_msg=True)
        (
            self.model,
            _,
            _,
            self.current_epoch,
            _,
        ) = load_checkpoint(
            model=self.model,
            optimizer=None,
            lr_scheduler=None,
            checkpoint_path=checkpoint_path,
        )
        # print(len(self.train_dataset.places), len(self.val_dataset.places), len(self.test_dataset.places))
        # print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
        worst_acuracy = self.run_an_epoch_with_group(self.test_loader, epoch=0, mode=Mode.test)  #
        change_column_value_of_existing_row(
            "accuracy",
            worst_acuracy,
            self.run_configs_file_path,
            self.run_id,
        )
