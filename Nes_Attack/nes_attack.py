import argparse
import os
# import glob
# import glog as log
import sys
import torch
import yaml
import time

project_path = '/root/myProject/HOTCOLDBlock-main-V3/'
os.chdir(project_path)
sys.path.append(project_path)

from victim_detector.utils.general import (check_dataset, check_img_size,check_yaml, colorstr,intersect_dicts,scale_coords,
                                           xywh2xyxy,non_max_suppression)
from victim_detector.models.yolo import Model
from class_NES import NES
from config import CLASS_NUM, OUT_DIR
from victim_detector.utils.torch_utils import de_parallel




def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.print('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def get_exp_dir_name(dataset):
    dirname = 'NES-attack-{}'.format(dataset)
    return dirname

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


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
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--samples-per-draw', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=1e-3, help="Sampling variance.")
    parser.add_argument('--epsilon', type=float, default=4.6) #跟随源代码中ImageNet的参数
    parser.add_argument('--log-iters', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max-queries', type=int, default=100000)
    parser.add_argument('--save-iters', type=int, default=50)
    parser.add_argument('--plateau-drop', type=float, default=2.0)
    parser.add_argument('--min-lr-ratio', type=int, default=200)
    parser.add_argument('--plateau-length', type=int, default=5)
    parser.add_argument('--imagenet-path', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max-lr', type=float, default=2.0)#跟随源代码中ImageNet的参数
    parser.add_argument('--min-lr', type=float, default=5e-5)
    # PARTIAL INFORMATION ARGUMENTS
    parser.add_argument('--top-k', type=int, default=-1, help="if you don't want to use the partial information mode, "
                                                              "just leave this argument to -1 as the default setting."
                                                              "Note that top-k must be set to true class number in the targeted attack.")
    parser.add_argument('--adv-thresh', type=float, default=-1.0)
    # LABEL ONLY ARGUMENTS
    parser.add_argument('--label-only', action='store_true', help="still on developing in progress")
    parser.add_argument('--zero-iters', type=int, default=100,
                        help="how many points to use for the proxy score, which is still on developing")
    parser.add_argument('--label-only-sigma', type=float, default=1e-3,
                        help="distribution width for proxy score, which is still on developing")

    parser.add_argument('--starting-eps', type=float, default=None)
    parser.add_argument('--starting-delta-eps', type=float, default=None)
    parser.add_argument('--min-delta-eps', type=float, default=None)
    parser.add_argument('--conservative', type=int, default=2,
                        help="How conservative we should be in epsilon decay; increase if no convergence")
    parser.add_argument('--exp-dir', default='data/output_allHigh_v3/data_kaist_nes04/', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--data', type=str, default='/root/myProject/HOTCOLDBlock-main-V3/data/input_v3/dataKaistAllHighPre/kaist.yaml', help='dataset.yaml path')
    parser.add_argument('--dataset_name', type=str, default='KAIST', help='dataset.yaml path')

    # parser.add_argument('--dataset', type=str, default="CIFAR-10",
    #                     choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
    #                     help='which dataset to use')
    parser.add_argument('--norm', type=str, default="l2", choices=["linf", "l2"],
                        help='Which lp constraint to update image [linf|l2]')
    # parser.add_argument('--arch', default="vgg19_bn", type=str, help='network architecture')
    # parser.add_argument('--test_archs', action="store_true")
    # parser.add_argument('--targeted', action="store_true")
    # parser.add_argument('--target_type', type=str, default='increment', choices=["random", "least_likely", "increment"])
    # parser.add_argument('--json_config', type=str,
    #                     default='/media/cqnu/4T-Disk/Wfs/Project/ASimulatorAttack-master/configures/NES_attack_conf.json',
    #                     help='a configures file to be passed in instead of arguments')
    # parser.add_argument("--total-images", type=int, default=1000)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images_5000 in "ram" (default) or "disk"')
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')

    # parser.add_argument('--attack_defense', action="store_true")
    # parser.add_argument('--defense_model', type=str, default=None)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()
    args.exp_dir = os.path.join(args.exp_dir, get_exp_dir_name(args.dataset_name))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    device = torch.device("cuda:0")
    log_file_path = os.path.join(args.exp_dir, 'run_{}.log'.format(args.dataset_name))
    set_log_file(log_file_path)

    log = Logger(OUT_DIR)

    log.print('Command line is: {}'.format(' '.join(sys.argv)))
    log.print("using GPU {}".format(args.gpu))
    log.print("Log file is written in {}".format(os.path.join(args.exp_dir, 'run.log')))
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
    hyp['cls'] *= CLASS_NUM[args.dataset_name] / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = args.label_smoothing

    model.hyp = hyp  # attach hyperparameters to model


    ###########################

    attacker = NES(args, gs, hyp, imgsz, log)
    save_result_path = args.exp_dir + "/{}_result.json".format(args.dataset_name)
    log.print("Begin attack yolov5 on {}, result will be saved to {}".format(args.dataset_name, save_result_path))
    attacker.attack_all_images(args, args.dataset_name, model, save_result_path)
    model.cpu()
    log.print("All done!")