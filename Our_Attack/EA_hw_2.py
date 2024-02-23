import torch
import yaml
import argparse
import gc
from tqdm import tqdm
import numpy as np
import os
import sys
from pathlib import Path

#筛选图片在预训练的yolov5上的识别率

project_path = '/root/myProject/HOTCOLDBlock-main-V3/'
os.chdir(project_path)
sys.path.append(project_path)

from victim_detector.models.yolo import Model
from victim_detector.utils.general import intersect_dicts, check_img_size, non_max_suppression,scale_coords, xywh2xyxy, check_dataset
# from victim_detector.utils.datasets import create_dataloader
from target_model.yolov3.utils.datasets import create_dataloader
from victim_detector.utils.metrics import box_iou
from Our_attack.pso_EA_hw_2 import PSO,OptimizeFunction
from config import IMG_NAMES

from utils_patch import *
from target_model.yolov3.utils.torch_utils import get_targets_conf_noP




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
    iou = box_iou(labels[:, 1:], detections[:, :4]) #所有预测box与所有真实box的交并比
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().detach().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct
def get_imgs_mAP(img, targets, model, shapes, args, stats, seen):

    nb, _, height, width = img.shape
    with torch.no_grad():
        out, train_out = model(img)  # inference, loss outputs
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

def ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir='.', OUT_DIR = ".", names=(), eps=1e-16, tag = True):
# def ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

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
            plot_pr_curve(px, py, ap, 'data/output_allHigh_v3/data_flir02/PR_ori_curve.png', names)
        else:
            plot_pr_curve(px, py, ap, 'data/output_allHigh_v3/data_flir02/PR_adv_curve.png', names)



    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec
def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    #print('-------1------->', len(names))

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='victim_detector/models/yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--data', type=str, default='data/input/data_FLIR_highPreAll/flir_test_highPreAll.yaml', help='dataset.yaml path')
    parser.add_argument('--dataset_name', type=str, default='FLIR', help='dataset.yaml path')

    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.9, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    return parser.parse_args()

if __name__ == '__main__':

    device = torch.device("cuda")
    args = get_args()

    # Log
    OUT_DIR = 'data/output_allHigh_v3/data_flir02/'
    log = Logger(OUT_DIR)  # log file
    log.print("Args:")
    log.print(args)
    device = torch.device("cuda")

    #加载模型
    hyp = 'target_model/yolov3/runs/train/exp5-flir/hyp.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    weights = "target_model/yolov3/runs/train/exp5-flir/weights/best.pt"
    ckpt = torch.load(weights, map_location='cpu')
    model = Model(ckpt['model'].yaml, ch=3, nc=16, anchors=hyp.get('anchors')).to(device)
    exclude = ['anchor'] if (hyp.get('anchors')) and not resume else []
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    model.load_state_dict(csd, strict=False)
    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()


    #加载数据
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple


    data_dict = check_dataset(args.data)
    img_path = data_dict['val']
    dataloader, dataset = create_dataloader(img_path, imgsz, args.batch_size, gs, args)


    func = OptimizeFunction(model, device)  # 适应度函数
    round = 1
    succ_num = 0

    succ_query_list = []
    all_query_list = []
    attack_num = 0
    for i, (imgs, targets, paths, shape) in tqdm(enumerate(dataloader), total=len(dataloader)):

        torch.cuda.empty_cache()
        targets = targets.to(device)
        imgs = imgs.to(device, non_blocking=True).half() / 255  # (n, c, h, w)
        name = os.path.basename(paths[0])

        with torch.no_grad():
            ori_out, ori_train_out = model(imgs)
        ori_avg_conf= get_targets_conf_noP(ori_out, args.conf_thres, args.iou_thres)

        if ori_avg_conf == 0:
            log.print("image {} is adversary".format(i+1))
            continue

        if ori_avg_conf.item() > 0.5:
            attack_num += 1
            pso = PSO(100, imgs, model, device, log, i, args, name)  # 初始化50个具有对抗性的粒子
            if pso.del_tag:
                log.print("初始化失败，删除图片{}".format(name))
                continue

            func.set_para(targets, imgs)
            pso.optimize(func)
            success, succ_num, query_num, adv_img = pso.run(args, log, i, succ_num, name, imgs)


            stats_adv, seen_adv = get_imgs_mAP(adv_img, targets, model, shape, args, stats_adv, seen_adv)

            if success:
                succ_query_list.append(query_num)
                all_query_list.append(query_num)
            else:
                all_query_list.append(query_num)
        else:
            log.print("image {} is adversary".format(i))

    stats_adv = [np.concatenate(x, 0) for x in zip(*stats_adv)]
    data_adv = np.save(
        '/root/myProject/HOTCOLDBlock-main-V3/data/output_allHigh_v3/data_flir02/stats_adv_info', aa)
    if len(stats_adv) and stats_adv[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_adv, names=IMG_NAMES[args.dataset_name], tag=False)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, adv_map50, adv_map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats_adv[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    log.print(pf % ('all', seen, nt.sum(), mp, mr, adv_map50, adv_map))


    log.print(succ_query_list)
    log.print(all_query_list)

    mean_query = sum(succ_query_list) / len(succ_query_list)
    mean_query_all = sum(all_query_list) / len(all_query_list)
    median_query = get_median(succ_query_list)
    median_query_all = get_median(all_query_list)

    log.print("攻击成功率为：{:.2f}, 查询均值为：{:.2f}/{:.2f}, 查询中值为：{:.2f}/{:.2f}".format(succ_num / attack_num, mean_query,
                                                                               mean_query_all, median_query,
                                                                               median_query_all))
