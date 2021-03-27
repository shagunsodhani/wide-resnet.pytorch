import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import xplogger
import xplogger.logbook
from omegaconf import OmegaConf

import config as cf
from networks import *

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning_rate")
parser.add_argument("--net_type", default="wide-resnet", type=str, help="model")
parser.add_argument("--depth", default=28, type=int, help="depth of model")
parser.add_argument("--widen_factor", default=10, type=int, help="width of model")
parser.add_argument("--dropout", default=0.3, type=float, help="dropout_rate")
parser.add_argument(
    "--dataset", default="cifar10", type=str, help="dataset = [cifar10/cifar100]"
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--testOnly", "-t", action="store_true", help="Test mode with the saved model"
)
args = parser.parse_args()
args = OmegaConf.create(vars(args))
args.description = "testing xplogger/wandb"
args.id = "2"

logbook_config = xplogger.logbook.make_config(
    logger_dir=f"logs/{args.id}",
    wandb_config={
        "project": "codistillation",
        "entity": "sodhani",
        "save_code": True,
        "config": OmegaConf.to_container(args, resolve=True),
        "id": args.id,
    },
    wandb_prefix_key="mode",
)
logbook = xplogger.logbook.LogBook(config=logbook_config)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = (
    cf.start_epoch,
    cf.num_epochs,
    cf.batch_size,
    cf.optim_type,
)

# Data Uplaod
logbook.write_message("\n[Phase 1] : Data Preparation")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ]
)  # meanstd transformation

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ]
)

if args.dataset == "cifar10":
    logbook.write_message("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform_test
    )
    num_classes = 10
elif args.dataset == "cifar100":
    logbook.write_message("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=False, transform=transform_test
    )
    num_classes = 100

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# Return network & file name
def getNetwork(args):
    if args.net_type == "lenet":
        net = LeNet(num_classes)
        file_name = "lenet"
    elif args.net_type == "vggnet":
        net = VGG(args.depth, num_classes)
        file_name = "vgg-" + str(args.depth)
    elif args.net_type == "resnet":
        net = ResNet(args.depth, num_classes)
        file_name = "resnet-" + str(args.depth)
    elif args.net_type == "wide-resnet":
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = "wide-resnet-" + str(args.depth) + "x" + str(args.widen_factor)
    else:
        logbook.write_message(
            "Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet"
        )
        sys.exit(0)

    return net, file_name


def get_checkpoint_path(args):
    return f"checkpoint/{args.id}"


def get_checkpoint_file(args, file_name):
    return f"checkpoint/{args.id}/{file_name}.pt"


# Test only option
if args.testOnly:
    logbook.write_message("\n[Test Phase] : Model setup")
    assert os.path.isdir("checkpoint"), "Error: No checkpoint directory found!"
    _, file_name = getNetwork(args)
    checkpoint = torch.load(get_checkpoint_file(args, file_name))
    net = checkpoint["net"]

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.0 * correct / total
        logbook.write_message("| Test Result\tAcc@1: %.2f%%" % (acc))

    sys.exit(0)

# Model
logbook.write_message("\n[Phase 2] : Model setup")
if args.resume:
    # Load checkpoint
    logbook.write_message("| Resuming from checkpoint...")
    assert os.path.isdir("checkpoint"), "Error: No checkpoint directory found!"
    _, file_name = getNetwork(args)
    checkpoint = torch.load(get_checkpoint_file(args, file_name))
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
else:
    logbook.write_message("| Building net type [" + args.net_type + "]...")
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(
        net.parameters(),
        lr=cf.learning_rate(args.lr, epoch),
        momentum=0.9,
        weight_decay=5e-4,
    )

    logbook.write_message(
        f"Training Epoch {epoch}, LR {cf.learning_rate(args.lr, epoch)}"
    )
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        outputs = net(inputs)  # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # sys.stdout.write("\r")
        # sys.stdout.write(
        #     "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%"
        #     % (
        #         epoch,
        #         num_epochs,
        #         batch_idx + 1,
        #         (len(trainset) // batch_size) + 1,
        #         loss.item(),
        #         100.0 * correct / total,
        #     )
        # )
        logbook.write_metric(
            {
                "epoch": epoch,
                "iter": batch_idx + 1,
                "loss": loss.item(),
                "acc@1": 100.0 * correct.item() / total,
                "mode": "train",
            }
        )
        # sys.stdout.flush()


def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.0 * correct / total
        logbook.write_message(
            "\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%"
            % (epoch, loss.item(), acc.item())
        )
        logbook.write_metric(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "acc@1": acc.item(),
                "mode": "validation",
            }
        )

        if acc > best_acc:
            logbook.write_message("| Saving Best model...\t\t\tTop1 = %.2f%%" % (acc))
            state = {
                "net": net.module if use_cuda else net,
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir(get_checkpoint_path(args)):
                os.mkdir(get_checkpoint_path(args))
            save_point = "./checkpoint/" + args.dataset + os.sep + args.id + os.sep

            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, get_checkpoint_file(args, file_name))
            best_acc = acc


logbook.write_message("Phase 3 : Training model")
logbook.write_message("| Training Epochs = " + str(num_epochs))
logbook.write_message("| Initial Learning Rate = " + str(args.lr))
logbook.write_message("| Optimizer = " + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    logbook.write_message("| Elapsed time : %d:%02d:%02d" % (cf.get_hms(elapsed_time)))

logbook.write_message("\n[Phase 4] : Testing model")
logbook.write_message("* Test results : Acc@1 = %.2f%%" % (best_acc))

