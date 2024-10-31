import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
from ecfd.gaussian import *
from models.resnet import *
from models.senet import *
from models.mobilenet import *
from models.mobilenetv2 import *
from models.shufflenet import *
from models.shufflenetv2 import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1027)
torch.cuda.manual_seed(1027)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--class_num', default=100, type=int)
parser.add_argument('--epoch', default=160, type=int)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)
parser.add_argument('--optim', default=True, type=bool)
parser.add_argument('--optimize_sigma', default=False, type=bool)


args = parser.parse_args()
print(args)


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


BATCH_SIZE = 128
LR = 0.1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset, testset = None, None
if args.class_num == 100:
    print("dataset: CIFAR100")
    trainset = torchvision.datasets.CIFAR100(
        root='/home/zhanglf/kd_aug/data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='/home/zhanglf/kd_aug/data',
        train=False,
        download=True,
        transform=transform_test
    )
if args.class_num == 10:
    print("dataset: CIFAR10")
    trainset = torchvision.datasets.CIFAR10(
        root='/home/zhanglf/data',
        train=True,
        download=False,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='/home/zhanglf/data',
        train=False,
        download=False,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

if args.model == "resnet18":
    model_name = resnet18
    feat_size = 512
if args.model == "resnet50":
    model_name = resnet50
    feat_size = 2048
if args.model == "resnet101":
    model_name = resnet101
    feat_size = 2048
if args.model == "resnet152":
    model_name = resnet152
    feat_size = 2048
if args.model == 'wideresnet50':
    model_name = wide_resnet50_2
    feat_size = 2048
if args.model == "resnext50":
    model_name = resnext50_32x4d
    feat_size = 2048
if args.model == "mobilenet":
    model_name = mobilenet
    feat_size = 1024
if args.model == "mobilenetv2":
    model_name = mobilenetv2
    feat_size = 1280
if args.model == "shufflenet":
    model_name = shufflenet
    feat_size = 960
if args.model == "shufflenetv2":
    model_name = shufflenetv2
    feat_size = 1024
if args.model == "preactresnet18":
    model_name = preactresnet18
    LR /= 5
    feat_size = 512
if args.model == "preactresnet50":
    model_name = preactresnet50
    LR /= 5
    feat_size = 2048
if args.model == "senet18":
    model_name = seresnet18
    feat_size = 512
if args.model == "senet50":
    model_name = seresnet50
    feat_size = 2048



teacher = resnext50_32x4d()
teacher.adaptation_layers = nn.ModuleList([
        nn.Linear(2048, 512),
        nn.Linear(2048, 512),
        nn.Linear(2048, 512),
        nn.Linear(2048, 512)
    ])
#   teacher adaptation layers are not used in training. We add them just because the teacher in .pth has them.
teacher.load_state_dict(torch.load("/home/zhanglf/kd-benchmarks/resnext_teacher.pth"))
teacher.to(device)

net = model_name()
if args.optim:
    net.adaptation_layer = nn.Linear(feat_size, 2048)
else:
    teacher.adaptation_layer = nn.Linear(feat_size, 2048)
    teacher.cuda()
net.to(device)
criterion = nn.CrossEntropyLoss()
# optimize_sigma
if args.optimize_sigma:
    print('optimize sigma')
    lg_sigmas = torch.zeros(1, 2048).cuda()
    lg_sigmas.requires_grad = True
    param = list(net.parameters()) + [lg_sigmas]
else:
    param = net.parameters()
    lg_sigmas = [1.0]
# define optimizer
optimizer = optim.SGD(
    [
        {'params': param},
    ], lr=LR, weight_decay=5e-4, momentum=0.9,
)

acc1 = 0


def frequency_loss(s_feat, t_feat, lg_sigmas, optimize_sigma):
    # s_feat: batchsize x hidden state (128 x 2048)
    # t_feat: batchsize x hidden state (128 x 2048)
    # to do: define the frequency function loss here to replace the following loss
    if optimize_sigma:
        sigmas = torch.exp(lg_sigmas)
    else:
        sigmas = lg_sigmas
    ecfd_loss = gaussian_ecfd(s_feat, t_feat, sigmas, optimize_sigma)
    # loss = torch.dist(s_feat, t_feat) * 1e-2
    return ecfd_loss


with torch.no_grad():
    correct = 0.0
    total = 0.0
    for data in testloader:
        teacher.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = teacher(images)
        _, predicted = torch.max(outputs.data, 1)
        total += float(labels.size(0))
        correct += float((predicted == labels).sum())
    acc1 = (100 * correct / total)
    print('teacher accuracy is', acc1)

    
if __name__ == "__main__":
    best_acc = 0
    print("Start Training")
    for epoch in range(args.epoch):
        if epoch in [60, 120, 150]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        teacher.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        sum_distill_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, feature_list = net(inputs)
            b = labels.size(0)
            with torch.no_grad():
                t_outputs, t_feature_list = teacher(inputs)

            loss = criterion(outputs, labels)
            t_feat = torch.nn.functional.adaptive_avg_pool2d(t_feature_list[-1], 1).view(b, -1)
            s_feat = torch.nn.functional.adaptive_avg_pool2d(feature_list[-1], 1).view(b, -1)
            if args.optim:
                s_feat = net.adaptation_layer(s_feat)
            else:
                s_feat = teacher.adaptation_layer(s_feat)
            distill_loss = CrossEntropy(outputs, t_outputs)
            distill_loss += frequency_loss(s_feat, t_feat, lg_sigmas, optimize_sigma=args.optimize_sigma)

            sum_distill_loss += float(distill_loss)
            loss += distill_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(labels.size(0))
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            if i % 20 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f, %.03f  Acc: %.4f%%'
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), sum_distill_loss / (i + 1),
                         100 * correct / total))
        print("Waiting Test!")

        acc1 = 0
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += float(labels.size(0))
                correct += float((predicted == labels).sum())
            acc1 = (100 * correct / total)
            if acc1 > best_acc:
                best_acc = acc1
                #torch.save(net.state_dict(), "./checkpoints/" + args.model+"_fitnet.pth")
        print('Test Set Accuracy: %.4f%%' % acc1)
    print("Training Finished, TotalEPOCH=%d" % args.epoch)
    print ("Highest Accuracy is ", best_acc)

