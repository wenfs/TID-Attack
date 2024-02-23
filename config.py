# CLASS_NUM = {"FLIR":16,"CVC09-D":1, "CVC09-N":1, "KAIST":4}
CLASS_NUM = {"MNIST":10,"FashionMNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10, "TinyImageNet":200}
# IMGSZ = {"FLIR":(640, 512),"CVC09-D":(640, 480), "CVC09-N":(640, 480), "KAIST":(640, 512)}
IMAGE_SIZE = {"CIFAR-10":(32,32), "CIFAR-100":(32,32), "ImageNet":(224,224), "MNIST":(28, 28), "FashionMNIST":(28,28), "SVHN":(32,32),
              "TinyImageNet": (64,64)}
PY_ROOT = "/root/myProject/HOTCOLDBlock-main-V3"
# IN_CHANNELS = {"FLIR":3,"CVC09-D":3, "CVC09-N":3, "KAIST":3}
IN_CHANNELS = {"MNIST":1, "FashionMNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3, "TinyImageNet":3}

OUT_DIR = '/root/myProject/HOTCOLDBlock-main-V3/data/output_allHigh_v3/data_kaist_nes04/'
IMG_NAMES = {"FLIR":{0: "person", 1: "bike", 2: "car", 3: "motor", 4: "bus", 5: "train", 6: "truck", 7: "light",
                 8: "hydrant", 9: "sign", 10: "dog", 11: "deer", 12: "skateboard", 13: "stroller", 14: "scooter",
                 15: "other vehicle"},
             "CVC09-D":{0:"person"},
             "CVC09-N":{0:"person"},
             "KAIST":{0:"person", 1:"people", 2:"cyclist", 3:"person?"}}

IMAGE_DATA_ROOT = {"CIFAR-10":"/home1/machen/dataset/CIFAR-10"}


pretrained_cifar_model_conf = {"CIFAR-10":{
                                "vgg11_bn":None,
                                "vgg13_bn":None,
                                "vgg16_bn":None,
                                "vgg19_bn":None,
                                "alexnet":None,
                                "alexnet_bn":None,
                                "resnet-20":{"depth":20, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-32":{"depth":32, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-44":{"depth":44, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-50":{"depth":50, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-56":{"depth":56, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "resnet-110":{"depth":110, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "resnet-1202":{"depth":1202,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "preresnet-110":{"depth":110,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4,  "block_name":"BasicBlock"},
                                "resnext-8x64d":{"depth":29, "cardinality":8, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1, "drop":0},
                                "resnext-16x64d":{"depth":29, "cardinality":16, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1,  "drop":0},
                                "WRN-28-10":{"depth":28, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-28-10-drop":{"depth":28, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10":{"depth":34, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10-drop":{"depth":34, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10-drop":{"depth":40, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10":{"depth":40, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "densenet-bc-100-12":{"depth":100,"growthRate":12,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2,"drop":0},
                                "densenet-bc-L190-k40":{"depth":190,"growthRate":40,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2,"drop":0},
                                 "pcl_resnet-110": {"depth":110},
                                "pcl_resnet-50": {"depth":50}
                            },
                            "CIFAR-100":{
                                "vgg11_bn":None,
                                "vgg13_bn":None,
                                "vgg16_bn":None,
                                "vgg19_bn":None,
                                "alexnet":None,
                                "alexnet_bn":None,
                                "resnet-20":{"depth":20, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-32":{"depth":32, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-44":{"depth":44, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-50":{"depth":50, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-56":{"depth":56, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-110":{"depth":110, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "resnet-1202":{"depth":1202,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "preresnet-110":{"depth":110,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4,  "block_name":"BasicBlock"},
                                "resnext-8x64d":{"depth":29, "cardinality":8, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1, "drop":0},
                                "resnext-16x64d":{"depth":29, "cardinality":16, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1, "drop":0},
                                "WRN-28-10":{"depth":28, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-28-10-drop":{"depth":28, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10":{"depth":34, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10-drop":{"depth":34, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10-drop":{"depth":40, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10":{"depth":40, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "densenet-bc-100-12":{"depth":100,"growthRate":12,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2, "drop":0},
                                "densenet-bc-L190-k40":{"depth":190,"growthRate":40,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2, "drop":0},
                                "pcl_resnet-110": {"depth": 110},
                                "pcl_resnet-50": {"depth": 50}
                            },
                            "TinyImageNet":{
                                "pcl_resnet110": {"depth":110},
                                "pcl_resnet50": {"depth":50}
                            }
            }


