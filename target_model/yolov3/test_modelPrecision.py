import torch
from tqdm import tqdm
import numpy as np
import yaml
import argparse



from utils_patch import *
from target_model.yolov3.models.experimental import attempt_load
from target_model.yolov3.utils.general import check_img_size
from target_model.yolov3.utils.datasets import create_dataloader
from victim_detector.utils.general import check_dataset,non_max_suppression,scale_coords, xywh2xyxy
from victim_detector.utils.metrics import ap_per_class, box_iou

project_path = '/media/cqnu/4T-Disk/Wfs/Project/HOTCOLDBlock-main'
os.chdir(project_path)
sys.path.append(project_path)

device = torch.device("cuda")
iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='victim_detector/models/yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--data', type=str, default='data_coco2017_val/coco_val2017.yaml', help='dataset.yaml path')
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    return parser.parse_args()

def get_imgs_mAP(img, targets, model, shapes, args, stats, seen):

    nb, _, height, width = img.shape
    out, train_out = model(img)  # inference, loss outputs
    # NMS
    targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
    out = non_max_suppression(out, args.conf_thres, args.iou_thres, multi_label=True)  # (300,6)

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
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

img_names = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane', 5:'bus', 6:'train', 7:'truck', 8:'boat', 9:'traffic light',
        10:'fire hydrant', 11:'stop sign', 12:'parking meter', 13:'bench', 14:'bird', 15:'cat', 16:'dog', 17:'horse', 18:'sheep', 19:'cow',
        20:'elephant', 21:'bear', 22:'zebra', 23:'giraffe', 24:'backpack', 25:'umbrella', 26:'handbag', 27:'tie', 28:'suitcase', 29:'frisbee',
        30:'skis', 31:'snowboard', 32:'sports ball', 33:'kite', 34:'baseball bat', 35:'baseball glove', 36:'skateboard', 37:'surfboard',
        38:'tennis racket', 39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon', 45:'bowl', 46:'banana', 47:'apple',
        48:'sandwich', 49:'orange', 50:'broccoli', 51:'carrot', 52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair', 57:'couch',
        58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet', 62:'tv', 63:'laptop', 64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone',
        68:'microwave', 69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book', 74:'clock', 75:'vase', 76:'scissors', 77:'teddy bear',
        78:'hair drier', 79:'toothbrush'}
s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
print(s)

args = get_args()

#model
hyp = 'target_model/yolov3/yolov3.yaml'
if isinstance(hyp, str):
    with open(hyp, errors='ignore') as f:  # f中包含了配置文件.yaml中的所有内容
        hyp = yaml.safe_load(f)  # hyp返回python字典，通过get()方法调取其中的参数
weights = "target_model/yolov3/weights/yolov3.pt"
model = attempt_load(weights, map_location=device).to(device)  # load FP32 model
stride = int(model.stride.max())  # model stride
gs = max(int(model.stride.max()), 32)  # grid size (max stride)
imgsz = check_img_size(args.imgsz, s=stride)  # check img_size
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()  # to FP16
model.eval()
#data
data_dict = check_dataset(args.data)
img_path = data_dict['val']
dataloader, dataset = create_dataloader(img_path,imgsz,args.batch_size, gs, args)

stats_clean = []
seen = 0
# 计算干净样本的mAP,单张单张地计算
for i, (imgs, targets, paths, shapes) in tqdm(enumerate(dataloader), total=len(dataloader)):

    torch.cuda.empty_cache()
    ##
    targets = targets.to(device)
    ##################
    # d_min = torch.min(imgs)
    # d_max = torch.max(imgs)
    # a = 0 +(1-0)/(d_max-d_min) * (imgs - d_min)
    ##################

    imgs = imgs.to(device, non_blocking=True).half() / 255  # (n, c, h, w)
    stats_clean, seen = get_imgs_mAP(imgs, targets, model, shapes, args, stats_clean, seen)
stats_clean = [np.concatenate(x, 0) for x in zip(*stats_clean)]

if len(stats_clean) and stats_clean[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_clean, names=img_names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, clean_map50, clean_map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats_clean[3].astype(np.int64), minlength=80)  # number of targets per class
else:
    nt = torch.zeros(1)
# Print results
pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
print(pf % ('all', seen, nt.sum(), mp, mr, clean_map50, clean_map))

               # Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95
               #   all         98        638      0.693      0.638      0.699      0.567