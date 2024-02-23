import argparse
import glob
import os
import random
import os.path as osp
import json
import numpy as np
import glog as log
import sys
import time
from tqdm import tqdm
import torch
import yaml

from types import SimpleNamespace

project_path = '/root/myProject/HOTCOLDBlock-main-V3'
os.chdir(project_path)
sys.path.append(project_path)

from victim_detector.models.yolo import Model
from victim_detector.utils.general import (check_dataset, check_img_size,check_yaml, colorstr,intersect_dicts,scale_coords,
                                           xywh2xyxy,non_max_suppression)
from target_model.yolov3.utils.datasets import create_dataloader

from ZOO_attack.zoo_attack import ZOOAttack
from config import PY_ROOT, CLASS_NUM, IMGSZ, IMG_NAMES
from victim_detector.utils.metrics import box_iou,compute_ap, plot_pr_curve
from victim_detector.utils.torch_utils import de_parallel


device = torch.device("cuda:0")
jdict, stats_clean, stats_adv_simba_flir, ap, ap_class = [], [], [], [], []
iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()  # 通过numel()函数，我们可以迅速查看一个张量到底又多少元素。
conf_thres = float(0.25)  # confidence threshold
iou_thres = 0.9
single_cls = False,  # treat as single-class dataset
dt, p, r, f1, mp, mr, clean_map50, clean_map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
adv_map50 = 0.0
adv_map = 0.0
seen = 0
seen_adv = 0


def ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir='.', OUT_DIR = ".", names=(), eps=1e-16, tag = True):
    # Sort by objectness
    i = np.argsort(-conf) #返回的是有小到大排序后的数字下标
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i] #从小到大排序

    # Find unique classes，unique_classes为类别，nt为对应的类别数量
    unique_classes, nt = np.unique(target_cls, return_counts=True) #对于一维数组或者列表，np.unique() 函数 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表。
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels_5000
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs，cumsum返回给定axis上的累计和
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        # plot_pr_curve(px, py, ap, Path(save_dir) / 'plots/PR_curve.png', names)
        if tag:
            plot_pr_curve(px, py, ap, 'data/output_allHigh_v3/data_cvcN_zooV301/PR_ori_curve.png', names)
        else:
            plot_pr_curve(px, py, ap, 'data/output_allHigh_v3/data_cvcN_zooV301/PR_adv_curve.png', names)

        # plot_mc_curve(px, f1, 'data/output/debug/plots/F1_curve.png', names, ylabel='F1')
        # plot_mc_curve(px, p, 'data/output/debug/plots/P_curve.png', names, ylabel='Precision')
        # plot_mc_curve(px, r, 'data/output/debug/plots/R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).detach().cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct
def get_imgs_mAP(img, targets, target_model, shapes, args, stats, seen):

    nb, _, height, width = img.shape
    with torch.no_grad():
        out, train_out = target_model(img)  # inference, loss outputs
    # NMS
    targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
    out = non_max_suppression(out, 0.25, 0.45, multi_label=True)  # (300,6)

    # Metrics
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]  # target的zhenshilabel，包含bbox信息，取出来原始label中的第一列（类别）
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        shape = shapes[si][0]
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        predn = pred.clone()  # shape=(300,6)
        scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred,对predn更改

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels_5000，对tbox更改
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels_5000,将原始标签的第二列与tbox合并
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

    return  stats, seen

class ZooAttackFramework(object):

    def __init__(self, dataloader):
        self.dataset_loader = dataloader
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        # self.not_done_prob_all = torch.zeros_like(self.query_all)


    def make_adversarial_examples(self, batch_index, images,  args, shapes, targets, target_model, attacker, log, stats_adv_simba_flir, seen_adv):
        # target_model = attacker.model
        batch_size = images.size(0)
        selected = torch.arange(batch_index * batch_size, min((batch_index + 1) * batch_size, self.total_images))  # 选择这个batch的所有图片的index
        target_labels = None
        log.print("Begin attack batch {}!".format(batch_index))
        with torch.no_grad():
            adv_images, stats_info= attacker.attack(images, targets, log, batch_index)

        stats_adv_simba_flir, seen_adv = get_imgs_mAP(adv_images.half(), targets, target_model, shapes, args,
                                                      stats_adv_simba_flir, seen_adv)

        query = stats_info["query"]
        correct = stats_info["correct"]
        not_done = stats_info["not_done"]
        success = stats_info["success"]
        success_query = stats_info["success_query"]
        # not_done_prob = stats_info["not_done_prob"]
        not_done_loss = stats_info["not_done_loss"]
        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_loss']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

        return stats_adv_simba_flir, seen_adv

    def attack_dataset_images(self, args, arch_name, target_model, result_dump_path, log):
        # attacker = ZOOAttack(target_model, IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0], args)
        attacker = ZOOAttack(target_model, 3, IMGSZ[args.dataset][0], args, log)

        ##############################################33
        stats_clean = []
        seen = 0
        # 记录攻击之前干净样本的精度
        for i, (imgs_clean, targets_clean, paths, shapes) in tqdm(enumerate(self.dataset_loader),
                                                                  total=len(self.dataset_loader)):
            torch.cuda.empty_cache()
            targets_clean = targets_clean.to(device)
            imgs_clean = imgs_clean.to(device, non_blocking=True).half() / 255  # (n, c, h, w)
            stats_clean, seen = get_imgs_mAP(imgs_clean, targets_clean, target_model, shapes, args, stats_clean, seen)

        stats_clean = [np.concatenate(x, 0) for x in zip(*stats_clean)]
        bb = np.array(stats_clean)
        # 保存stats信息
        data = np.save(
            '/root/myProject/HOTCOLDBlock-main-V3/data/output_allHigh_v3/data_cvcN_zooV301/stats_ori_info', bb)
        if len(stats_clean) and stats_clean[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_clean, OUT_DIR, names=IMG_NAMES[args.dataset],
                                                          tag=True)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, clean_map50, clean_map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats_clean[3].astype(np.int64),
                             minlength=CLASS_NUM[args.dataset])  # number of targets per class
        else:
            nt = torch.zeros(1)
        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        log.print(pf % ('all', seen, nt.sum(), mp, mr, clean_map50, clean_map))
        ##############################################33


        # for batch_idx, data_tuple in enumerate(self.dataset_loader):
        stats_adv_simba_flir = []
        seen_adv = 0
        for batch_idx, (imgs, targets, paths, shapes) in tqdm(enumerate(self.dataset_loader),total=len(self.dataset_loader)):

            im = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            im = im.float().half()  # uint8 to fp16/32
            im /= 255
            with torch.no_grad():
                stats_adv_simba_flir, seen_adv = self.make_adversarial_examples(batch_idx, im.cuda(),  args, shapes, targets, target_model,  attacker, log,
                                                                                stats_adv_simba_flir, seen_adv)

        ##################################
        stats_adv_simba_flir = [np.concatenate(x, 0) for x in zip(*stats_adv_simba_flir)]
        aa = np.array(stats_adv_simba_flir)
        data_adv = np.save(
            '/root/myProject/HOTCOLDBlock-main-V3/data/output_allHigh_v3/data_cvcN_zooV301/stats_adv_info', aa)

        if len(stats_adv_simba_flir) and stats_adv_simba_flir[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_adv_simba_flir, names=IMG_NAMES[args.dataset],
                                                          tag=False)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, adv_map50, adv_map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats_adv_simba_flir[3].astype(np.int64),
                             minlength=CLASS_NUM[args.dataset])  # number of targets per class
        else:
            nt = torch.zeros(1)
        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        log.print(pf % ('all', seen, nt.sum(), mp, mr, adv_map50, adv_map))
        ##################################
        log.print('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.print('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.print('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.print(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.print(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.print('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        if self.not_done_all.sum().item() > 0:
            log.print(
                '  avg not_done_loss: {:.4f}'.format(self.not_done_loss_all[self.not_done_all.byte()].mean().item()))
            # log.print(
            #     '  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.print("所有查询为：")
        log.print(self.query_all)

        log.print('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          # "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          'args': vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.print("done, write stats info to {}".format(result_dump_path))




def get_exp_dir_name(dataset, use_tanh, use_log, use_uniform_pick, targeted, target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    use_tanh_str = "tanh" if use_tanh else "no_tanh"
    use_log_str = "log_softmax" if use_log else "no_log"
    randomly_pick_coordinate = "randomly_sample" if use_uniform_pick else "importance_sample"
    dirname = 'ZOO-{}-{}-{}-{}-{}'.format(dataset, use_tanh_str, use_log_str, target_str, randomly_pick_coordinate)
    return dirname

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.print('{:s}: {}'.format(prefix, args.__getattribute__(key)))



def main(args, arch, model, dataloader, log):

    # if args.init_size is None:
    #     args.init_size = model.input_size[-1]
    #     log.print("Argument init_size is not set and not using autoencoder, set to image original size:{}".format(
    #         args.init_size))

    target_str = "untargeted" if not args.targeted else "targeted_{}".format(args.target_type)
    save_result_path = args.exp_dir + "/data_{}@arch_{}@solver_{}@{}_result.json".format(args.dataset,
                                                                                            arch, args.solver,
                                                                                            target_str)
    if os.path.exists(save_result_path):
        model.cpu()
        return
    attack_framework = ZooAttackFramework(dataloader)
    attack_framework.attack_dataset_images(args, arch, model, save_result_path, log)
    model.cpu()

class Logger:
    def __init__(self, file, print=True):
        self.file = file
        local_time = time.strftime("%b%d_%H%M%S", time.localtime())
        self.file += local_time
        self.All_file = 'logs/All.log'

    def print(self, content='', end='\n', file=None):
        if file is None:
            file = self.file
        with open(file, 'a') as f:
            if isinstance(content, str):
                f.write(content + end)
            else:
                old = sys.stdout
                sys.stdout = f
                print(content)
                sys.stdout = old
        if file is None:
            self.print(content, file=self.All_file)
        print(content, end=end)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="the batch size for zoo, zoo_ae attack")
    parser.add_argument("-c", "--init_const", type=float, default=0.0,
                        help="the initial setting of the constant lambda")
    parser.add_argument("-d", "--dataset", type=str, default="KAIST")
    parser.add_argument('--data', type=str, default='data/input_v3/dataKaistAllHighPre/kaist.yaml', help='dataset.yaml path')

    # choices=["CIFAR-10", "CIFAR-100", "TinyImageNet", "ImageNet", "MNIST", "FashionMNIST"])
    parser.add_argument("-m", "--max_iterations", type=int, default=10000, help="set 0 to use the default value")
    parser.add_argument("-p", "--print_every", type=int, default=10,
                        help="print information every PRINT_EVERY iterations")
    parser.add_argument("--binary_steps", type=int, default=1)
    parser.add_argument("--targeted",help="the type of attack")
    parser.add_argument("--target_type", type=str, default="increment", choices=['random', 'least_likely', "increment"],
                        help="if set, choose random target, otherwise attack every possible target class, only works when using targeted")
    parser.add_argument("-o", "--early_stop_iters", type=int, default=100,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--abort_early", action="store_true")
    parser.add_argument("--init_size", default=None, type=int, help="starting with this size when --use_resize")
    parser.add_argument("--resize", action="store_true",
                        help="this option only works for the preprocess resize of images")

    parser.add_argument("-r", "--reset_adam", action='store_true', help="reset adam after an initial solution is found")
    parser.add_argument("--solver", choices=["adam", "newton", "adam_newton", "fake_zero"], default="adam")
    parser.add_argument('--test_archs')
    parser.add_argument('--arch', type=str, default="yolov3", help='network architecture')
    parser.add_argument('--exp_dir', default='data/output_allHigh_v3/data_kaist_zooV301/', type=str,
                        help='directory to save results and logs')
    parser.add_argument("--start_iter", default=0, type=int,
                        help="iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--use_tanh")
    parser.add_argument("--use_log")
    parser.add_argument("--epsilone", type=float, default=45)
    parser.add_argument('--seed', default=1216, type=int, help='random seed')
    # parser.add_argument('--json_config', type=str,
    #                     default='/media/cqnu/4T-Disk/Wfs/Project/ASimulatorAttack-master/configures/zoo_attack_conf.json',
    #                     help='a configures file to be passed in instead of arguments')
    parser.add_argument("--uniform", action='store_true', help="disable importance sampling")

    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                        help='--cache images_5000 in "ram" (default) or "disk"')
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.use_tanh, args.use_log, args.uniform,args.targeted, args.target_type))
    os.makedirs(args.exp_dir, exist_ok=True)
    set_log_file(os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch)))
    OUT_DIR = "data/output_allHigh_v3/data_kaist_zooV301/"
    log = Logger(OUT_DIR)

    args.abort_early = True
    if args.init_const == 0.0:
        if args.binary_steps != 0:
            args.init_const = 0.01
        else:
            args.init_const = 0.5
    random.seed(args.seed)
    np.random.seed(args.seed)

    archs = []
    dataset = args.dataset
    archs.append(args.arch)
    log.print('Command line is: {}'.format(' '.join(sys.argv)))
    log.print("Log file is written in {}".format(osp.join(args.exp_dir, 'run.log')))
    log.print('Called with args:')
    print_args(args)

    #加载模型
    hyp = 'target_model/yolov3/runs/train/exp2-Kaist-100/hyp.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:  # f中包含了配置文件.yaml中的所有内容
            hyp = yaml.safe_load(f)  # hyp返回python字典，通过get()方法调取其中的参数
    # coco数据集上的预训练权重
    # weights = "runs/train/exp/weights/best.pt"
    weights = "target_model/yolov3/runs/train/exp2-Kaist-100/weights/best.pt"
    ckpt = torch.load(weights, map_location='cpu')
    model = Model(ckpt['model'].yaml, ch=3, nc=4, anchors=hyp.get('anchors')).to(device)
    # exclude = ['anchor'] if (hyp.get('anchors')) and not args.resume else []
    exclude = ['anchor'] if (hyp.get('anchors')) and not resume else []
    csd = ckpt['model'].float().state_dict()  # state_dict变量存放训练过程中需要学习的权重和偏执系数
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    model.load_state_dict(csd, strict=False)
    model.half()
    model.eval()
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    ###########################
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / args.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= args.batch_size * accumulate / nbs  # scale weight_decay
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= CLASS_NUM[args.dataset] / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = args.label_smoothing

    model.hyp = hyp  # attach hyperparameters to model

    ###########################

    #加载数据
    data_dict = check_dataset(args.data)
    path = data_dict['val']
    dataloader = create_dataloader(path,
                                   imgsz,
                                   # batch_size // WORLD_SIZE * 2,
                                   1,
                                   gs,
                                   args,
                                   hyp=hyp,
                                   cache=None if args.noval else args.cache,
                                   rect=True,
                                   rank=-1,
                                   workers=args.workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]

    ##attack
    for arch in archs:
        main(args, arch, model, dataloader, log)



