import copy
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as TF
# from utils_patch import PatchApplier_v1
from utils_patch import PatchApplier
# from ptop import ParticleToPatch_One
from ptop import ParticleToPatch
from  target_model.yolov3.utils.torch_utils import  *
from skimage.io import imsave
from skimage import img_as_ubyte



class OptimizeFunction:
    def __init__(self, detector, device):
        self.detector = detector
        self.device = device
        self.size = 0
        self.num_patch = 1
        # self.patch_size = patch_size
        
    def set_para(self, targets, imgs):
        self.targets = targets
        self.imgs = imgs



    def evaluate(self, x, args):
        with torch.no_grad():
            succ, score = get_adv_adjudge_noP(x[1], self.detector, args)
            #适应度函数
            if succ :
                return_obj_loss = 1*50 - get_l2_distance(x[1], x[0], norm=2)

            else:
                return_obj_loss = 1*0 - get_l2_distance(x[1], x[0], norm=2)


        return return_obj_loss
        

class SwarmParameters:
    pass


class Particle:
    def __init__(self, imgs, model, args, log, device):
        self.device = device
        self.model = model
        self.dimensions = imgs.shape
        self.out_tag = False
        self.history_value = 0

        classes = 2
        self.nQuery = 0
        

        x_adv = imgs.clone()
        with torch.no_grad():
            ori_out, ori_train_out = model(imgs)

        ori_obj_loss = get_targets_conf_noP(ori_out, args.conf_thres, args.iou_thres)

        best_loss = ori_obj_loss
        iter = 0
        while True:
            iter += 1
            print("初始化step_{}".format(iter))
            random_noise = torch.tensor(np.random.normal(0, 0.01 ** 0.5, x_adv.shape)).to(device) #随机高斯噪音
            x_adv_noi = torch.from_numpy(np.clip(((x_adv + random_noise).detach().cpu().numpy()),0.0,1.0)).half()
            x_adv_noi = x_adv_noi.to(self.device)
            #step1:判断当前随机的噪音是否需要加上去
            with torch.no_grad():
                noi_out, noi_train_out = model(x_adv_noi)

            noi_obj_loss = get_targets_conf_noP(noi_out, args.conf_thres, args.iou_thres)
            if noi_obj_loss < best_loss:
                # step2:判断加上了之后是否攻击成功
                best_loss = noi_obj_loss
                with torch.no_grad():
                    delta, delta_score = get_adv_adjudge_noP(x_adv_noi, self.model, args)
                x_adv = x_adv_noi
                if delta: #当前随机出来的噪音有效
                    break
                else:
                    continue
            else:
                x_adv = x_adv

            if iter > 50000:
                self.out_tag = True
                break
            else:
                self.out_tag = False


        if self.out_tag == False:
            print("当前粒子初始化完成")
            self.position = [imgs, x_adv]
            self.velocity = torch.zeros(imgs.shape).to(self.device)

            self.pbest_position = self.position
            self.pbest_value = torch.Tensor([float("-inf")]).to(self.device) #距离/损失
        else:
            print("当前粒子初始化失败")
            return

    
    def update_velocity(self, gbest_position, victory_tag, w, c1, c2, c3, c4, imgs):

        z = torch.randn(1).to(self.device)

        if victory_tag: #快速阶段
            self.velocity = w * self.velocity \
                                   + c1 * z * (self.pbest_position[1] - self.position[1]) \
                                   + c2 * z * (gbest_position[1] - self.position[1]) \

        else: #稳定阶段
            self.velocity = w * self.velocity \
                            + c2 * z * (gbest_position[1] - self.position[1]) \


        return
        
        
    def move(self):
        self.position[1] = (self.position[1] + self.velocity).half()

        self.position[1].data.clamp_(0.0,1.0)
        self.position[0].data.clamp_(0,1)

    def rollback(self,history,i):

        if len(history) > 1:
            self.position = history[0]['pos'][i]
            self.velocity = history[0]['vel'][i]


def add_noise_to_bbox(image, bbox):
    img = image.clone()
    img = torch.squeeze(img).detach().cpu().numpy().transpose(2,1,0)
    img_h, img_w, _ = img.shape
    #在每个person的bbox区域添加扰动
    for k in range(len(bbox)):
        x1, y1, x2, y2 = bbox[k]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox_h, bbox_w = y2 - y1, x2 - x1
        crop_img = img[y1:y2, x1:x2, :]
        noise = np.random.normal(0, 0.01 ** 0.5, crop_img.shape)
        noisy_img = crop_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        image_with_noise = img.copy()
        image_with_noise[y1:y2, x1:x2, :] = noisy_img
    image_with_noise = torch.unsqueeze(torch.tensor(image_with_noise.transpose(2,1,0)), 0).half()
    return image_with_noise
        

class PSO:
    def __init__(self, swarm_size, imgs, model, device, log, index, args, name):
        self.max_iterations = 3
        self.swarm_size = swarm_size
        self.model = model
        self.ori_img = imgs

        self.gbest_position = [0, 0]
        self.gbest_particle = None
        self.gbest_dist = torch.Tensor([float("inf")]).to(device)
        self.gbest_value = torch.Tensor([float("-inf")]).to(device) #距离/损失
        self.swarm = []
        self.nQuery = 0  # 初始化时要耗费6次查询
        self.history = []
        #
        self.w = 0.175
        self.c1 = 0.42
        self.c2 = 1.4
        self.c3 = 0.004
        self.c4 = 0.0035
        self.del_tag = False


        print("image{}__{}初始化开始".format(index+1, name))

        for i in range(self.swarm_size): #初始化3个粒子,通过对原始图像添加随机噪声生成初始图像粒子群
            self.swarm.append(Particle(imgs, self.model, args, log, device=device))
            for k in self.swarm:
                if k.out_tag == True:
                    self.del_tag = True
            if self.del_tag:
                break

        if self.del_tag:
            print("image{}__{}初始化失败".format(index + 1, name))
        else:
            print("image{}__{}初始化成功".format(index + 1, name))


    
    def optimize(self, function):
        self.fitness_function = function
        
        
    def run(self, args, log, i, succ_num, name, imgs):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0
        round = 1
        fitness_value = 0
        swarm_dist = 0
        num = 0

        for iter in range(int(100000/3)):
            torch.cuda.empty_cache()
            #速度更新公式中的所有权重随着迭代次数的增加而减小
            self.c1  = self.c1 - (4 - 0)*iter/int(100000/3)
            self.c2  = self.c2 - (4 - 0)*iter/int(100000/3)
            self.c3  = self.c3 - (4 - 0)*iter/int(100000/3)
            self.c4  = self.c4 - (4 - 0)*iter/int(100000/3)
            if iter == 0:
                # no_confloss_tag = True
                for particle in self.swarm:
                    self.nQuery += particle.nQuery
                    l2_dist = get_l2_distance(particle.position[1], particle.position[0], norm=2)
                    fitness_value += 1*50 - l2_dist
                    fit_val = 50 - l2_dist
                    particle.history_value = fit_val
                    swarm_dist += l2_dist
                    if (particle.pbest_value < fit_val):
                        particle.pbest_value = fit_val  # best_value值就是损失值
                        particle.pbest_position[0] = particle.position[0].clone()
                        particle.pbest_position[1] = particle.position[1].clone()
                swarm_dist = swarm_dist / self.swarm_size
                _, init_delta_conf = get_adv_adjudge_noP(self.ori_img, self.model, args)
                init_fitness_value = fitness_value/self.swarm_size
                log.print('image %d: initial state. query = %d, dist = %.4f,delta_confidenceOfOrigin = %.4f, fitness_value = %.4f' % (
                        i + 1, self.nQuery, swarm_dist, init_delta_conf, init_fitness_value))

                for particle in self.swarm:
                    best_fit_cand = self.fitness_function.evaluate(particle.position, args)
                    if self.gbest_value < best_fit_cand:
                        self.gbest_value = best_fit_cand
                        self.gbest_position[0] = particle.position[0].clone()
                        self.gbest_position[1] = particle.position[1].clone()
                        self.gbest_particle = copy.deepcopy(particle)  # 深拷贝，会拷贝对象及其子对象
                temp_val = self.gbest_value


            if fitness_value < 10 or init_fitness_value < 10:
                victory_tag = True
                if round <= 16:
                    self.history.append({"pos": [copy.deepcopy(p.pbest_position) for p in self.swarm],
                                         "vel": [copy.deepcopy(p.velocity) for p in self.swarm]})
                    if len(self.history) > 1:
                        del self.history[1]
                else:
                    for k, particle in enumerate(self.swarm):
                        particle.rollback(self.history, k)
            else:
                victory_tag = False
            #局部极值
            for particle in self.swarm:
                particle.update_velocity(self.gbest_position, victory_tag, self.w, self.c1, self.c2, self.c3,self.c4, imgs)  # 计算速度
                particle.move()
                fitness_candidate = self.fitness_function.evaluate(particle.position, args)

                #满足条件则执行柯西变异
                r3 = np.random.rand(1)
                if iter/100000 <= 0.5 and r3 > 0.9:
                    for kk in range(20):
                        cauthy = np.random.standard_cauchy(1)
                        cauthy = torch.tensor(0.5 + np.arctan(cauthy)/np.pi).cuda()
                        tem_position = particle.pbest_position[1] + cauthy * torch.tensor(np.ones(particle.pbest_position[1].shape)).cuda()
                        tem_partical = [particle.position[0], tem_position.half()]

                        cur_fit = self.fitness_function.evaluate(tem_partical, args)
                        if fitness_candidate.item() <  cur_fit.item():
                            fitness_candidate  = cur_fit
                            particle.position[1] = tem_partical[1]
                            break
                        else:
                            continue


                vel = np.linalg.norm(particle.velocity.detach().cpu().numpy())
                self.w = torch.sigmoid(0.01 * ((fitness_candidate - particle.history_value)/vel))
                particle.history_value = fitness_candidate

                self.nQuery += 1
                if (particle.pbest_value < fitness_candidate):
                    particle.pbest_value = fitness_candidate  # best_value值就是损失值
                    particle.pbest_position[0] = particle.position[0].clone()
                    particle.pbest_position[1] = particle.position[1].clone()
            # --- Set GBest 全局极值
            pa_val = 0
            k = 0
            for particle in self.swarm:
                best_fitness_candidate= self.fitness_function.evaluate(particle.position,  args)
                pa_val += particle.position[1]
                self.nQuery += 1
                k += 1
                if self.gbest_value < best_fitness_candidate:
                    self.gbest_value = best_fitness_candidate
                    self.gbest_position[0] = particle.position[0]
                    self.gbest_position[1] = particle.position[1]



            #判断当前的全局最优是否满足条件
            with torch.no_grad():
                attack_tag, delta_conf = get_adv_adjudge_noP(self.gbest_position[1], self.model, args)
            dist = get_l2_distance(self.gbest_position[1], self.gbest_position[0], norm=2)
            if attack_tag and dist < 45 and self.nQuery <= 100000:
                success = True
                swarm_parameters.gbest_position = self.gbest_position
                swarm_parameters.best_distance = dist
                log.print('image %d，iteration:%d: attack is successful. query = %d, dist = %.4f, delta_confidence = %.4f, fitness_value = %.4f' % (
                    i + 1, iter, self.nQuery, dist, delta_conf, self.gbest_value))
                succ_num += 1
                # 保存对抗样本和噪音
                adv_img = self.gbest_position[1].squeeze(0).cpu().detach().numpy().clip(0, 1).transpose((1, 2, 0))
                noise = np.uint8((self.gbest_position[1] - self.gbest_position[0]).squeeze(0).cpu().detach().numpy()).transpose((1, 2, 0))

                imsave('data/output_allHigh/data_flir01/adv_imgs/' + name, img_as_ubyte(adv_img))  # Save adversarial example
                imsave('data/output_allHigh/data_flir01/noises/' + name, noise)  # Save adversarial example

                break
            else:
                success = False
                log.print('image %d，iteration:%d: attack is not successful. query = %d, dist = %.4f, delta_confidence = %.4f, fitness_value = %.4f' % (
                    i + 1, iter, self.nQuery, dist, delta_conf, self.gbest_value))
                if self.nQuery > 100000:
                     log.print(
                        'image %d，iteration:%d: attack is not successful. query = %d, dist = %.4f, delta_confidence = %.4f, fitness_value = %.4f' % (
                            i + 1, iter, self.nQuery, dist, delta_conf, self.gbest_value))
                    break

                continue

        return success, succ_num, self.nQuery, self.gbest_position[1]