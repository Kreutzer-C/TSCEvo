import argparse
import sys
import logging
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import clip
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from batchgenerators.utilities.file_and_folder_operations import *
from utils.loss import *
from utils.co_module import do_collaboration
from data import data_helper
from data.data_info import get_data_info
from model.submodel import Classifier
from model.backbone_factory import get_backbone


def get_args():
    parser = argparse.ArgumentParser(description="Teacher-Student Co-Evolution for Source-Free Domain Adaptation with Vision-Language Model")
    # Experiment Name
    # (Determines where the results are saved, highly recommended to keep it different for each experiment)
    parser.add_argument("--exp", type=str, default="TSCEvo")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    # Backbone Network Setting (Only the types below are supported)
    # --backbone = resnet18 | resnet50 | resnet 101
    # --CLIP_backbone = ViT-B/16 | ViT-B/32
    parser.add_argument("--backbone", default="resnet50", help="target model arch")
    parser.add_argument("--CLIP_backbone", default="ViT-B/16", help="CLIP model vision encoder arch")

    # LoRA Setting
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='list of attention matrices where deploy LoRA')
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate applied before the LoRA module')

    # Dataset Setting (Only --dataset needs to be determined, the others will be configured automatically)
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")

    # Training Setting (two learning rates may need to be adjusted depending on the specific task)
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate_tar", "-lr1", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--learning_rate_lora", "-lr2", type=float, default=5e-5, help="Learning rate")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")

    # Trade-off Parameters Setting
    parser.add_argument("--lamb1", default=1.0, type=float,
                        help='trade-off scaler between absolute textsim loss and relative textsim loss')
    parser.add_argument("--lamb2", default=1.0, type=float,
                        help='trade-off scaler between clip loss and textsim loss')
    parser.add_argument("--lamb3", default=10.0, type=float,
                        help='trade-off scaler of L_cls in L_KT')
    parser.add_argument("--lamb4", default=0.5, type=float,
                        help='trade-off scaler of L_dist in L_KT')
    parser.add_argument("--alpha_decay", "-ad", default=0.5, type=float,
                        help='trade-off scaler of memory bank updating')

    return parser.parse_args()


def get_clip_outputs(clip_model, image, prompt):
    with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
        image_feature = clip_model.encode_image(image)

        classify_prompt = [cp.replace('_', ' ') for cp in prompt]
        classify_token = clip.tokenize(classify_prompt).to(image.device)
        text_feature = clip_model.encode_text(classify_token)

    image_feature_norm = image_feature / image_feature.norm(dim=-1, keepdim=True)
    text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.data
    logit_scale = logit_scale.exp()
    outputs = (logit_scale * image_feature_norm @ text_feature_norm.T).to(torch.float32)
    return outputs


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2
        self.lamb3 = args.lamb3
        self.lamb4 = args.lamb4
        self.clip_feature_dim = 512  # For ViT-B/16 or ViT-B/32

        self.clip_model, _ = clip.load(self.args.CLIP_backbone, device=self.device, download_root='./pretrain/CLIP')
        self.list_lora_layers = apply_lora(args, self.clip_model)
        if args.dataset == 'Terra':
            load_lora(args, self.list_lora_layers, load_path=args.lora_cp_path)
        self.clip_model = self.clip_model.to(device)
        print(len(self.list_lora_layers))
        mark_only_lora_as_trainable(self.clip_model)

        self.featurizer = get_backbone(args.backbone).to(device)
        self.classifier = Classifier(self.clip_feature_dim, args.n_classes, is_nonlinear=False).to(device)
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            self.featurizer,
            self.classifier
        )
        self.model.load_state_dict(torch.load(args.checkpoint_path))
        logging.info(f">>>Loading model from: {args.checkpoint_path}")

        self.cpmb = torch.full((args.n_classes, args.n_classes), 1 / args.n_classes).to(device)

        self.classify_prompt = [f"a photo of a {c.replace('_', ' ')}" for c in self.args.classes]

        self.target_loader = data_helper.get_val_dataloader(args)
        logging.info("Dataset size: OD test(target) %d" % (len(self.target_loader.dataset)))

        self.optimizer_tar = optim.SGD(self.model.parameters(), lr=args.learning_rate_tar,
                                       momentum=0.9, nesterov=True, dampening=0.0)
        self.scheduler_tar = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_tar,
                                                                  args.epochs, eta_min=1e-3 * args.learning_rate_tar)
        self.optimizer_lora = optim.AdamW(get_lora_parameters(self.clip_model), lr=args.learning_rate_lora,
                                          betas=(0.9, 0.999))
        self.scheduler_lora = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_lora,
                                                                   args.epochs, eta_min=1e-3 * args.learning_rate_lora)

        self.cosine_sim_criterion = nn.CosineEmbeddingLoss()

        self.best_epoch_acc = 0.0
        self.writer = SummaryWriter(args.output_folder + '/log')


    def _do_epoch(self):
        scaler = torch.cuda.amp.GradScaler()

        # Stage1:
        logging.info("\nDoing CLIP LoRA Tuning...")
        correct_predictions = 0
        correct_predictions_mix = 0
        total_samples = 0
        for iteration, ((data, class_l), d_idx) in enumerate(tqdm(self.target_loader)):
            data, class_l = data.to(self.device), class_l.to(self.device)

            self.clip_model.eval()
            self.model.eval()
            with torch.no_grad():
                z = self.featurizer(data)
                p_tar = torch.softmax(self.classifier(z), dim=1)
                out_clip = get_clip_outputs(self.clip_model, data, self.classify_prompt)
                p_clip = torch.softmax(out_clip, dim=1)
                p_mix, self.cpmb = do_collaboration(p_tar, p_clip, self.cpmb, self.args.alpha)

            y_pred_mix = torch.argmax(p_mix, dim=1)
            anchor_prompt = [f"a {self.args.classes[i].replace('_',' ')}" for i in y_pred_mix]
            entangled_prompts = []
            for i in y_pred_mix:
                cls_name = self.args.classes[i].replace('_', ' ')
                single_entangled_prompt = [f"a {domain.replace('_', ' ')} of a {cls_name}" for domain in
                                           self.args.source]
                entangled_prompts.extend(single_entangled_prompt)  # [bs*3, 512]

            self.clip_model.train()
            self.optimizer_lora.zero_grad()
            with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
                image_feature = self.clip_model.encode_image(data)

                anchor_token = clip.tokenize(anchor_prompt).to(self.device)
                anchor_feature = self.clip_model.encode_text(anchor_token)  # [bs, 512]

                entangled_tokens = clip.tokenize(entangled_prompts).to(self.device)
                entangled_features = self.clip_model.encode_text(entangled_tokens)
                entangled_features = entangled_features.view(-1, len(self.args.source),
                                                             self.clip_feature_dim)  # [bs,3,512]

            ab_sim_loss = absolute_sim_loss(anchor_feature, entangled_features)
            re_sim_loss = relative_sim_loss(self.cosine_sim_criterion, entangled_features)
            loss_textsim = ab_sim_loss + self.lamb1 * re_sim_loss

            loss_clip = clip_contrastive_loss(image_feature, anchor_feature)

            loss_lora = loss_clip + self.lamb2 * loss_textsim

            scaler.scale(loss_lora).backward()
            scaler.step(self.optimizer_lora)
            scaler.update()

            self.writer.add_scalar('lora/loss_lora', loss_lora, iteration + self.current_epoch * len(self.target_loader))
            self.writer.add_scalar('lora/loss_clip', loss_clip, iteration + self.current_epoch * len(self.target_loader))
            self.writer.add_scalar('lora/loss_textsim', loss_textsim, iteration + self.current_epoch * len(self.target_loader))

            y_pred_clip = torch.argmax(p_clip, dim=1)
            correct_predictions += (y_pred_clip == class_l).sum().item()
            correct_predictions_mix += (y_pred_mix == class_l).sum().item()
            total_samples += class_l.size(0)

        clip_accuracy = correct_predictions / total_samples
        mix_accuracy = correct_predictions_mix / total_samples
        logging.info(f"\nCLIP Accuracy: {clip_accuracy:.4f} Mix Accuracy:{mix_accuracy:.4f}")
        self.writer.add_scalar('lora/clip_acc', clip_accuracy, self.current_epoch)
        self.writer.add_scalar('lora/mix_acc', mix_accuracy, self.current_epoch)

        # Stage2:
        logging.info("\nDoing Knowledge Transfer...")
        self.clip_model.eval()
        self.model.train()
        correct_predictions = 0
        total_samples = 0
        for iteration, ((data, class_l), d_idx) in enumerate(self.target_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            self.optimizer_tar.zero_grad()
            z = self.featurizer(data)
            out_tar = self.classifier(z)
            p_tar = torch.softmax(out_tar, dim=1)

            with torch.no_grad():
                out_clip = get_clip_outputs(self.clip_model, data, self.classify_prompt)
                p_clip = torch.softmax(out_clip, dim=1)

                p_mix, self.cpmb = do_collaboration(p_tar, p_clip, self.cpmb, self.args.alpha)
                y_pred_mix = torch.argmax(p_mix, dim=1)

                anchor_prompt = [f"a {self.args.classes[i].replace('_', ' ')}" for i in y_pred_mix]
                with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
                    anchor_token = clip.tokenize(anchor_prompt).to(self.device)
                    anchor_feature = self.clip_model.encode_text(anchor_token)

            loss_mi = IID_loss(p_tar, p_clip)
            loss_cls = (- p_mix * p_tar).sum(dim=1).mean()
            loss_dist = feature_dist_loss(anchor_feature.to(torch.float32), z)
            loss = loss_mi + self.args.lamb3 * loss_cls + self.args.lamb4 * loss_dist

            loss.backward()
            self.optimizer_tar.step()

            y_pred_tar = torch.argmax(p_tar, dim=1)
            correct_predictions += (y_pred_tar == class_l).sum().item()
            total_samples += class_l.size(0)

            if iteration % 10 == 0:
                logging.info("iter {}/{} loss: {:.6f} loss_mi: {:.6f} loss_cls: {:.6f} loss_dist: {:.6f}"
                             .format(iteration, len(self.target_loader), loss.item(), loss_mi.item(),
                                     loss_cls.item(), loss_dist.item()))

            self.iter_num = self.iter_num + 1
            self.writer.add_scalar('tar/loss', loss, self.iter_num)
            self.writer.add_scalar('tar/loss_mi', loss_mi, self.iter_num)
            self.writer.add_scalar('tar/loss_cls', loss_cls, self.iter_num)
            self.writer.add_scalar('tar/loss_dist', loss_dist, self.iter_num)

        tar_accuracy = correct_predictions / total_samples
        self.writer.add_scalar('tar/tar_acc', tar_accuracy, self.current_epoch)
        return tar_accuracy, clip_accuracy

    def do_training(self):
        self.iter_num = 0
        for self.current_epoch in tqdm(range(self.args.epochs)):
            tar_accuracy, clip_accuracy = self._do_epoch()
            self.scheduler_tar.step()
            self.scheduler_lora.step()

            if tar_accuracy >= self.best_epoch_acc:
                self.best_epoch_acc = tar_accuracy
                logging.info("\n*************NEW BEST!************")
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {tar_accuracy:.4f} CLIP: {clip_accuracy:.4f}\n')
                save_lora(self.args, self.list_lora_layers)
                target_save_path = join(self.args.output_folder, f'best_targetmodel.pth')
                torch.save(self.model.state_dict(), target_save_path)
            else:
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {tar_accuracy:.4f} CLIP: {clip_accuracy:.4f}\n')

    def do_test(self):
        logging.info("\n>>>=====================<<<")
        print("Testing on OD domain")
        logging.info(f'Domain: {self.args.target} Accuracy: {self.best_epoch_acc:.4f}')
        logging.info(">>>=====================<<<\n")


def train_TSCEvo():
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

        args.output_folder = join(os.getcwd(), 'results', args.exp, args.dataset, args.target)
        maybe_mkdir_p(args.output_folder)
        print(">>>output results will be at: {}".format(args.output_folder))

        args.checkpoint_path = join(os.getcwd(), 'pretrain', 'ERM', args.backbone, args.dataset,
                                    f"to_{args.target}_best.pth")
        args.lora_cp_path = join(os.getcwd(), 'pretrain', 'LORA', args.CLIP_backbone.replace('/', ''),
                                 args.dataset, args.target, 'lora.pt')

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
        logging.info(">>>Test on target domain:")
        logging.info(args.target)

        trainer = Trainer(args, device)
        trainer.do_training()
        trainer.do_test()


if __name__ == "__main__":
    train_TSCEvo()
