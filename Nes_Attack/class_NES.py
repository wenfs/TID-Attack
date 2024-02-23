import os
# import glog as log
import torch
import numpy as np
from torch.nn import functional as F
import json
from tqdm import tqdm


from config import CLASS_NUM,IMGSZ, OUT_DIR, IMG_NAMES
from victim_detector.utils.general import (check_dataset, check_img_size,check_yaml, colorstr,intersect_dicts,scale_coords,
                                           xywh2xyxy,non_max_suppression)
from target_model.yolov3.utils.datasets import create_dataloader
from victim_detector.models.yolo import Model
from victim_detector.utils.metrics import box_iou,compute_ap, plot_pr_curve
from victim_detector.utils.loss import ComputeLoss


device = torch.device("cuda:0")
jdict, stats_clean, stats_adv_simba_flir, ap, ap_class = [], [], [], [], []
iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()  # 通过numel()函数，我们可以迅速查看一个张量到底又多少元素。
conf_thres = float(0.25)  # confidence threshold
iou_thres = 0.45
single_cls = False,  # treat as single-class dataset
dt, p, r, f1, mp, mr, clean_map50, clean_map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

adv_map = 0.0
seen = 0
seen_adv = 0



class NES(object):
    def __init__(self, args, gs, hyp, imgsz, log):
        self.dataset_name = args.dataset_name
        self.num_classes = CLASS_NUM[self.dataset_name]
        self.dataset_loader = self.get_test_attacked_data(args, 1, gs, hyp, imgsz)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.log = log

    def get_test_attacked_data(self, args, batch_size, gs, hyp, imgsz):
        data_dict = check_dataset(args.data)
        path = data_dict['val']
        dataloader = create_dataloader(path,
                                       imgsz,
                                       # batch_size // WORLD_SIZE * 2,
                                       batch_size,
                                       gs,
                                       args,
                                       hyp=hyp,
                                       cache=None if args.noval else args.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=args.workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        return dataloader

    # def get_image_of_target_class(self,dataset_name, target_labels, target_model):
    #
    #     images = []
    #     for label in target_labels:  # length of target_labels is 1
    #         if dataset_name == "ImageNet":
    #             dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
    #         elif dataset_name == "CIFAR-10":
    #             dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
    #         elif dataset_name=="CIFAR-100":
    #             dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
    #
    #         index = np.random.randint(0, len(dataset))
    #         image, true_label = dataset[index]
    #         image = image.unsqueeze(0)
    #         if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
    #             image = F.interpolate(image,
    #                                    size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
    #                                    align_corners=False)
    #         with torch.no_grad():
    #             logits = target_model(image.cuda())
    #         while logits.max(1)[1].item() != label.item():
    #             index = np.random.randint(0, len(dataset))
    #             image, true_label = dataset[index]
    #             image = image.unsqueeze(0)
    #             if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
    #                 image = F.interpolate(image,
    #                                    size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
    #                                    align_corners=False)
    #             with torch.no_grad():
    #                 logits = target_model(image.cuda())
    #         assert true_label == label.item()
    #         images.append(torch.squeeze(image))
    #     return torch.stack(images) # B,C,H,W

    def xent_loss(self, logits, noise, true_labels, target_labels, top_k):
        if self.targeted:
            return F.cross_entropy(logits, target_labels, reduction='none'), noise  # FIXME 修改测试
        else:
            assert target_labels is None, "target label must set to None in untargeted attack"
            return F.cross_entropy(logits, true_labels, reduction='none'), noise

    def partial_info_loss(self, logits, noise, true_labels, target_labels, top_k):
        # logit 是融合了batch_size of noise 的, shape = (batch_size, num_classes)
        losses, noise = self.xent_loss(logits=logits,noise=noise, true_labels=true_labels, target_labels=target_labels, top_k=top_k)
        vals, inds = torch.topk(logits, dim=1, k=top_k, largest=True, sorted=True) # inds shape = (B, top_k)
        # inds is batch_size x k
        target_class = target_labels[0].item()  # 一个batch的target都是一样的
        good_image_inds = torch.sum(inds == target_class, dim=1).byte()    # shape = (batch_size,)
        losses = losses[good_image_inds]
        noise = noise[good_image_inds]
        return losses, noise

    #  STEP CONDITION (important for partial-info attacks)
    def robust_in_top_k(self, target_model, adv_images, target_labels, top_k):
        # 我自己增加的代码
        # if self.targeted:  # FIXME 作者默认targeted模式top_k < num_classes
        #     eval_logits = target_model(adv_images)
        #     t = target_labels[0].item()
        #     pred = eval_logits.max(1)[1][0].item()
        #     return pred == t
        if top_k == self.num_classes:   #
            return True
        eval_logits = target_model(adv_images)
        t = target_labels[0].item()
        _, top_pred_indices = torch.topk(eval_logits, k=top_k, largest=True,
                                               sorted=True)  # top_pred_indices shape = (1, top_k)
        top_pred_indices = top_pred_indices.view(-1).detach().cpu().numpy().tolist()
        if t not in top_pred_indices:
            return False
        return True

    def get_grad(self, x, sigma, samples_per_draw, batch_size, targets, target_model, loss_fn):
        num_batches = samples_per_draw // batch_size  # 一共产生多少个samples噪音点，每个batch
        losses = []
        grads = []

        for _ in range(num_batches):
            assert x.size(0) == 1
            noise_pos = torch.randn((batch_size//2,) + (x.size(1), x.size(2), x.size(3)))  # B//2, C, H, W
            noise = torch.cat([-noise_pos, noise_pos], dim=0).cuda()  # B, C, H, W
            eval_points = (x + sigma * noise).half()  # 1,C,H,W + B, C, H, W = B,C,H,W
            logits,logits_out = target_model(eval_points)  # B, num_classes

            # losses shape = (batch_size,)
            # if target_labels is not None:
            #     target_labels = target_labels.repeat(batch_size)
            # loss, noise = loss_fn(logits, noise, true_labels, target_labels, top_k)  # true_labels and target_labels have already repeated for batch_size
            # loss, loss_items = loss_fn(logits_out, targets)  # true_labels and target_labels have already repeated for batch_size
            loss = loss_fn(logits_out, targets)  # true_labels and target_labels have already repeated for batch_size
            # loss = - loss
            loss = loss.view(-1,1,1,1) # shape = (B,1,1,1)
            grad = torch.mean(loss * noise, dim=0, keepdim=True)/sigma # loss shape = (B,1,1,1) * (B,C,H,W), then mean axis= 0 ==> (1,C,H,W)
            losses.append(loss.mean())
            grads.append(grad)
        losses = torch.stack(losses).mean()  # (1,)
        grads = torch.mean(torch.stack(grads), dim=0)  # (1,C,H,W)
        return losses.item(), grads

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):
        stats_clean = []
        seen = 0
        #记录攻击之前干净样本的精度
        for i, (imgs_clean, targets_clean, paths, shapes) in tqdm(enumerate(self.dataset_loader), total=len(self.dataset_loader)):
            torch.cuda.empty_cache()
            targets_clean = targets_clean.to(device)
            imgs_clean = imgs_clean.to(device, non_blocking=True).half() / 255  # (n, c, h, w)
            stats_clean, seen = get_imgs_mAP(imgs_clean, targets_clean, target_model, shapes, args, stats_clean, seen)

        stats_clean = [np.concatenate(x, 0) for x in zip(*stats_clean)]
        bb = np.array(stats_clean)
        # 保存stats信息
        data = np.save('/root/myProject/HOTCOLDBlock-main-V3/data/output_allHigh_v3/data_kaist_nes04/stats_ori_info', bb)
        if len(stats_clean) and stats_clean[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_clean, OUT_DIR, names=IMG_NAMES[args.dataset_name], tag=True)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, clean_map50, clean_map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats_clean[3].astype(np.int64), minlength=CLASS_NUM[args.dataset_name])  # number of targets per class
        else:
            nt = torch.zeros(1)
        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        self.log.print(pf % ('all', seen, nt.sum(), mp, mr, clean_map50, clean_map))


        #执行NES攻击
        stats_adv_simba_flir = []
        seen_adv = 0
        for batch_idx, (imgs, targets, paths, shapes) in tqdm(enumerate(self.dataset_loader), total=len(self.dataset_loader)):

            fullname = os.path.basename(paths[0])
            (name, extention) = os.path.splitext(fullname)
            im = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            im = im.float().half()  # uint8 to fp16/32
            im /= 255

            with torch.no_grad():
                stats_adv_simba_flir, seen_adv = self.make_adversarial_examples(batch_idx, im.cuda(), args, targets, target_model, shapes, stats_adv_simba_flir, seen_adv)

        ##################################
        #
        stats_adv_simba_flir = [np.concatenate(x, 0) for x in zip(*stats_adv_simba_flir)]
        aa = np.array(stats_adv_simba_flir)
        data_adv = np.save('/root/myProject/HOTCOLDBlock-main-V3/data/output_allHigh_v3/data_kaist_nes04/stats_adv_info', aa)

        if len(stats_adv_simba_flir) and stats_adv_simba_flir[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_adv_simba_flir, names=IMG_NAMES[args.dataset_name], tag=False)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, adv_map50, adv_map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats_adv_simba_flir[3].astype(np.int64), minlength=CLASS_NUM[args.dataset_name])  # number of targets per class
        else:
            nt = torch.zeros(1)
        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        self.log.print(pf % ('all', seen, nt.sum(), mp, mr, adv_map50, adv_map))
        ##################################

        self.log.print('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        self.log.print('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        self.log.print('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            self.log.print('     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            self.log.print('   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            self.log.print('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        self.log.print('Saving results to {}'.format(result_dump_path))

        self.log.print("所有查询为：")
        self.log.print(self.query_all)

        query_all_np = self.query_all.detach().cpu().numpy().astype(np.int32)
        not_done_all_np = self.not_done_all.detach().cpu().numpy().astype(np.int32)
        correct_all_np = self.correct_all.detach().cpu().numpy().astype(np.int32)
        out_of_bound_indexes = np.where(query_all_np > args.max_queries)[0]
        if len(out_of_bound_indexes) > 0:
            not_done_all_np[out_of_bound_indexes] = 1
        success_all_np = (1 - not_done_all_np) * correct_all_np
        success_query_all_np = success_all_np * query_all_np
        success_indexes = np.nonzero(success_all_np)[0]

        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": np.mean(not_done_all_np[np.nonzero(correct_all_np)[0]]).item(),
                          "mean_query": np.mean(success_query_all_np[success_indexes]).item(),
                          "median_query": np.median(success_query_all_np[success_indexes]).item(),
                          "max_query": np.max(success_query_all_np[success_indexes]).item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": not_done_all_np.tolist(),
                          "query_all": query_all_np.tolist(),
                          'args': vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj,  sort_keys=True)
        self.log.print("done, write stats info to {}".format(result_dump_path))


    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_image_step(self, x, g, lr):

        return x + lr * g / self.norm(g)

    def linf_image_step(self, x, g, lr):
        if self.targeted:
            return x - lr * torch.sign(g)
        return x + lr * torch.sign(g)

    def l2_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.norm(delta) > eps).float()
            x = (orig + eps * delta / self.norm(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x
        return proj

    def proj(self, adv_imgs, orig_imgs, diff, max_distance):
        if adv_imgs.dim() == 3:
            adv_imgs = adv_imgs.unsqueeze(0)
            orig_imgs = orig_imgs.unsqueeze(0)
            diff = diff.unsqueeze(0)
        diff_norm = self.distance(torch.clamp(adv_imgs + diff, 0, 1), orig_imgs)
        factor = (max_distance / diff_norm).clamp(0, 1.0).reshape((-1, 1, 1, 1))
        adv_diff = (torch.clamp(adv_imgs + diff, 0, 1) - orig_imgs) * factor
        adv_imgs = torch.clamp(orig_imgs + adv_diff, 0, 1)
        return adv_imgs

    def distance(self, imgs1, imgs2=None, norm=2):
        # Compute L2 or L_inf distance between imgs1 and imgs2
        if imgs1.dim() == 3:
            imgs1 = imgs1.unsqueeze(0)
            imgs2 = imgs2.unsqueeze(0)
        img_num = imgs1.shape[0]
        if imgs2 is None:
            if norm == 2:
                distance = (imgs1.view([img_num, -1])).norm(2, dim=1)
                return distance
        if norm == 2:
            try:
                distance = (imgs1.view([img_num, -1]) - imgs2.view([img_num, -1])).norm(2, dim=1)
            except:
                print(img_num, imgs1.shape, imgs2.shape)
        elif norm == 'inf':
            distance = (imgs1.view([img_num, -1]) - imgs2.view([img_num, -1])).norm(float('inf'), dim=1)
        return distance

    def linf_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            return orig + torch.clamp(new_x - orig, -eps, eps)
        return proj

    def make_adversarial_examples(self, batch_index, images, args, targets, target_model, shapes, stats_adv_simba_flir, seen_adv):
        batch_size = args.batch_size  # Actually, the batch size of images is 1, the goal of args.batch_size is to sample noises
        # some statistics variables
        assert images.size(0) == 1

        with torch.no_grad():
            pred,_ = target_model(images)  # inference, loss outputs
        lb = []
        out = non_max_suppression(pred, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)[0]
        pre_conf = out[:, 4]
        after_selectConf_out = np.mean(pre_conf.detach().cpu().numpy())
        correct = True if after_selectConf_out >= 0.30 else False
        correct = torch.as_tensor(float(correct))
        # pred = logit.argmax(dim=1)
        query = torch.zeros(1).cuda()
        # correct = pred.eq(true_labels).float()  # shape = (1,)

        not_done = correct.clone()  # shape = (1,)
        success = (1 - not_done) * correct  # correct = 0 and 1-not_done = 1 --> success = 0
        success_query = success * query

        selected = torch.arange(batch_index, batch_index + 1)  # 选择这个batch的所有图片的index
        adv_images = images.clone()

        samples_per_draw = args.samples_per_draw  # samples per draw
        epsilon = args.epsilon   # 最终目标的epsilon
        goal_epsilon = epsilon
        max_lr = args.max_lr
        # ----- partial info params -----
        k = args.top_k
        adv_thresh = args.adv_thresh
        # if k > 0 or self.targeted:
        #     assert self.targeted, "Partial-information attack is a targeted attack."
        #     adv_images = self.get_image_of_target_class(self.dataset_name, target_labels, target_model)
        #     epsilon = args.starting_eps
        # else:   # if we don't want to top-k paritial attack set k = -1 as the default setting
        #
        k = self.num_classes
        delta_epsilon = args.starting_delta_eps
        g = torch.zeros_like(adv_images).cuda()
        last_ls = []
        # true_labels = true_labels.repeat(batch_size)  # for noise sampling points
        # max_iters = int(np.ceil(args.max_queries / args.samples_per_draw)) if k == self.num_classes else int(np.ceil(args.max_queries / (args.samples_per_draw + 1)))
        # if self.targeted:
        #     max_iters = int(np.ceil(args.max_queries / (args.samples_per_draw + 1)))
        # loss_fn = self.partial_info_loss if k < self.num_classes else self.xent_loss  # 若非paritial_information模式，k = num_classes
        loss_fn = ComputeLoss(target_model)  # 若非paritial_information模式，k = num_classes
        image_step = self.l2_image_step if args.norm == 'l2' else self.linf_image_step
        proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(images, args.epsilon)
        iter = 0
        while query[0].item() < args.max_queries:
            iter += 1
            # CHECK IF WE SHOULD STOP
            if not not_done.byte().any() and epsilon <= goal_epsilon:  # all success
                success_indicator_str = "success" if query[0].item() > 0 else "on a incorrectly classified image"
                self.log.print("Attack {} on {}-th image by using {} queries".format(success_indicator_str,
                                                                               batch_index, query[0].item()))
                #计算对抗样本的模型识别精度
                stats_adv_simba_flir, seen_adv = get_imgs_mAP(adv_images.half(), targets, target_model, shapes, args,
                                                              stats_adv_simba_flir, seen_adv)

                break

            prev_g = g.clone()
            l, g = self.get_grad(adv_images, args.sigma, samples_per_draw, batch_size, targets, target_model, loss_fn)
            query += samples_per_draw
            # log.info("Query :{}".format(query[0].item()))
            # SIMPLE MOMENTUM
            g = args.momentum * prev_g + (1.0 - args.momentum) * g
            # PLATEAU LR ANNEALING
            last_ls.append(l)
            last_ls = last_ls[-args.plateau_length:]  # FIXME 为何targeted的梯度会不断变大？
            condition = last_ls[-1] > last_ls[0] # if self.targeted else last_ls[-1] > last_ls[0]
            if condition and len(last_ls) == args.plateau_length:  # > 改成 < 号了调试，FIXME bug，原本的tf的版本里面loss不带正负号，如果loss变大，就降低lr
                if max_lr > args.min_lr:
                    max_lr = max(max_lr / args.plateau_drop, args.min_lr)
                    self.log.print("[log] Annealing max_lr : {:.5f}".format(max_lr))
                last_ls = []
            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr
            prop_de = 0.0
            # if l < adv_thresh and epsilon > goal_epsilon:
            if epsilon > goal_epsilon:
                prop_de = delta_epsilon

            while current_lr >= args.min_lr:
                # PARTIAL INFORMATION ONLY
                # if k < self.num_classes: #FIXME 我认为原作者写错了，这个地方改成targeted
                # GENERAL LINE SEARCH
                proposed_adv = image_step(adv_images, g, current_lr)
                # proposed_adv = proj_step(proposed_adv)
                proposed_adv = self.proj(proposed_adv,images,g,45)

                proposed_adv = torch.clamp(proposed_adv, 0, 1)
                if k != self.num_classes:
                    query += 1 # we must query for check robust_in_top_k
                if self.robust_in_top_k(target_model, proposed_adv, targets, k):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, args.min_delta_eps)
                        # delta_epsilon = prop_de
                    adv_images = proposed_adv
                    epsilon = max(epsilon - prop_de / args.conservative, goal_epsilon)
                    break
                elif current_lr >= args.min_lr * 2:
                    current_lr = current_lr / 2
                else:
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        break
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    self.log.print("[log] backtracking eps to {:.3f}".format(epsilon - prop_de,))


            with torch.no_grad():
                adv_logit,_ = target_model(adv_images.half())
            # adv_pred = adv_logit.argmax(dim=1)  # shape = (1, )
            lb = []
            adv_out = non_max_suppression(adv_logit, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)[0]
            adv_conf = adv_out[:, 4]
            after_selectConf_out = np.mean(adv_conf.detach().cpu().numpy())
            self.log.print("iter:{},  cur_loss:{},  cur_confidence:{},  cur_query:{}".format(iter, l, after_selectConf_out, query[0].item() ))

            not_done = True if after_selectConf_out >= 0.30 else False
            not_done = torch.as_tensor(float(not_done))
            # adv_prob = F.softmax(adv_logit, dim=1)
            # adv_loss, _ = loss_fn(adv_logit, None, true_labels[0].unsqueeze(0), target_labels, top_k=k)

            success = (1 - not_done) * correct * float(epsilon <= goal_epsilon)
            success_query = success * query
        else:
            self.log.print("Attack failed on {}-th image".format(batch_index))

        if epsilon > goal_epsilon:
            not_done.fill_(1.0)
            success.fill_(0.0)
            success_query = success * query

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

        return stats_adv_simba_flir, seen_adv


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
            plot_pr_curve(px, py, ap, 'data/output_allHigh_v3/data_kaist_nes04/PR_ori_curve.png', names)
        else:
            plot_pr_curve(px, py, ap, 'data/output_allHigh_v3/data_kaist_nes04/PR_adv_curve.png', names)

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