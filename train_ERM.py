import argparse
import os
import sys
import logging
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from batchgenerators.utilities.file_and_folder_operations import *
from data import data_helper
from model.submodel import Classifier
from model.backbone_factory import get_backbone
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="Source Domains ERM Pre-train")
    # Experiment Name
    # (Determines where the results are saved, highly recommended to keep it different for each experiment)
    parser.add_argument("--exp", type=str, default="Source_ERM")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    # Backbone Network Setting (Only the types below are supported)
    # --backbone = resnet18 | resnet50 | resnet 101
    parser.add_argument("--backbone", default="resnet50",
                        help="Which backbone network to use")

    # Dataset Setting (Only --dataset needs to be determined, the others will be configured automatically)
    parser.add_argument("--dataset", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--val_size", type=float, default=0.1)

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", "-l", type=float, default=5e-3, help="Learning rate")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.feature_dim = 512

        self.featurizer = get_backbone(args.backbone).to(device)
        self.classifier = Classifier(self.feature_dim, args.n_classes, is_nonlinear=False).to(device)
        self.model = nn.Sequential(
            self.featurizer,
            self.classifier
        )

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        logging.info("Dataset size: train %d, IID val %d" % (
            len(self.source_loader.dataset), len(self.val_loader.dataset)))

        self.optimizer = optim.SGD(self.model.parameters(), weight_decay=0.0005, momentum=0.9, lr=args.learning_rate)
        step_size = int(args.epochs * 0.8)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size)
        self.criterion = nn.CrossEntropyLoss()

        self.current_epoch = None

    def _do_epoch(self):
        self.model.train()
        for iter, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)

            # ERM
            loss = self.criterion(outputs, class_l)
            loss.backward()
            self.optimizer.step()

            if iter % 10 == 0:
                logging.info("iter {}/{} loss: {:.6f}".format(iter, len(self.source_loader), loss.item()))

        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        for iter, ((data, class_l), d_idx) in enumerate(self.val_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                outputs = self.model(data)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            correct_predictions += (pred == class_l).sum().item()
            total_samples += class_l.size(0)

        accuracy = correct_predictions / total_samples
        return accuracy


    def do_training(self):
        best_acc = 0.0
        for self.current_epoch in tqdm(range(self.args.epochs)):
            accuracy = self._do_epoch()
            self.scheduler.step()

            logging.info(f'Epoch: {self.current_epoch} IID Val Accuracy: {accuracy:.4f}')

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.output_folder, f"to_{self.args.target}_best.pth"))
                logging.info(f'NEW IID_best model checkpoint have been saved')


def ERM_pretrain():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:" + args.GPU_num if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        GPU_device = torch.cuda.get_device_properties(device)
        print(">>>===device:{}({},{}MB)".format(device, GPU_device.name, GPU_device.total_memory / 1024 ** 2))

    get_data_info(args)

    for domain in args.Domain_ID:
        args.target = domain
        args.source = args.Domain_ID.copy()
        args.source.remove(args.target)

        args.output_folder = os.path.join(os.getcwd(), 'results', args.exp, args.backbone, args.dataset, args.target)
        maybe_mkdir_p(args.output_folder)
        print(">>>output results will be saved at: {}".format(args.output_folder))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        logging.info("\n****************************")
        for key, value in vars(args).items():
            logging.info(f"{key}: {value}")
        logging.info("****************************\n")

        logging.info(">>>Training {} on source domains:".format(args.dataset))
        logging.info(args.source)

        trainer = Trainer(args, device)
        trainer.do_training()


if __name__ == "__main__":
    ERM_pretrain()