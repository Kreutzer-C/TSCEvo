"""
Test the OD performance of the source ERM pre-trained model, then register it for further SFDA task
"""

import argparse
import os
import sys
import shutil
import logging
import torch
from torch import nn
from batchgenerators.utilities.file_and_folder_operations import *
from data import data_helper
from model.submodel import Classifier
from model.backbone_factory import get_backbone
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="Source Domains ERM Pre-train")
    # Experiment Name
    # (The experiment name of ERM pre-train stage)
    parser.add_argument("--exp", type=str, required=True)

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

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

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
        self.checkpoint_path = join(self.args.output_folder, f'to_{args.target}_best.pth')
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        logging.info(f">>>Loading model from: {self.checkpoint_path}")

        self.target_loader = data_helper.get_val_dataloader(args)
        logging.info("Dataset size: OD test %d" % (len(self.target_loader.dataset)))

    def do_test(self):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        for iter, ((data, class_l), d_idx) in enumerate(self.target_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                outputs = self.model(data)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            correct_predictions += (pred == class_l).sum().item()
            total_samples += class_l.size(0)

        accuracy = correct_predictions / total_samples
        logging.info(f'Task: {self.args.source} --> {self.args.target} Accuracy: {accuracy:.4f}')
        logging.info(">>>=====================<<<\n")

    def do_regis(self):
        copy_target_path = join(os.getcwd(), 'pretrain', 'ERM', self.args.backbone, self.args.dataset)
        maybe_mkdir_p(copy_target_path)
        shutil.copy(self.checkpoint_path,
                    join(copy_target_path, f'to_{self.args.target}_best.pth'))


def ERM_test():
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
        if not os.path.exists(args.output_folder):
            raise ValueError(f"Path does not exist: {args.output_folder}, Please first run train_ERM.py")
        else:
            print(">>>using ERM pre-train output results saved at: {}".format(args.output_folder))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        logging.info("\n****************************")
        for key, value in vars(args).items():
            logging.info(f"{key}: {value}")
        logging.info("****************************\n")

        logging.info(">>>Test on target domain:")
        logging.info(args.target)

        trainer = Trainer(args, device)
        trainer.do_test()
        trainer.do_regis()


if __name__ == "__main__":
    ERM_test()
