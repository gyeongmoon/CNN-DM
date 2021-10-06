import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import argparse
import numpy as np
from model import *


############
# Variables.
parser = argparse.ArgumentParser(description='Train the AlexNet model for Continual Learning.')

parser.add_argument('--cnn_type', default='alexnet',  # ['resnet101', 'alexnet', 'vgg16', 'pre_alex_flowers', 'vgg_flowers', 'pre_alex_imagenet']
                    help='The type of CNN')
parser.add_argument('--dataset', default=['flowers'],  # ['imagenet', 'flowers', 'flowers_small', 'birds', 'scenes']
                    help='Dataset')
parser.add_argument('--num_classes', default=[102],  # [2, 102, 200, 67, 1000] -> It should be the last task's num_classes.
                    help='The number of layers')
parser.add_argument('--batch_size', default=30,  # [30, 40, 80, 256], [20] for LwF, retroModel.
                    help='The size of batch')
parser.add_argument('--num_epoch', default=100,
                    help='The size of epoch')  # [AlexNet, VGG-16, ResNet-101] -> [100, 60, 60] is used.
parser.add_argument('--seed', default=4,  # 4, 5, 8
                    help='Seed number')
parser.add_argument('--save', default=True,
                    help='The flag whether saving model or not')
parser.add_argument('--reuse', default='baseModel_flowers_alexnet',  # pre_alex_imagenet_flowers_model
                    help='The name of saved model')  # mem_flowers_model, KD_flowers_model, mem_imagenet_model, ...
parser.add_argument('--is_training', default=True,
                    help='[True, False] == [Training, Test]')

parser.add_argument('--lr', default=0.001,
                    help='The learning rate')  # [transfer, continual] -> [0.001, 0.0006] is used.
parser.add_argument('--momentum', default=0.9,
                    help='Momentum for SGD')
parser.add_argument('--weight_decay', default=0.0005,
                    help='Regularization [default=0.0005].')
parser.add_argument('--step_size', default=25,
                    help='Step size to decrease learning rate [default=40].')  # epoch=[60, 100] -> [25, 40].

parser.add_argument('--is_fine_tuning', default=False,
                    help='[True, False] == [Fine_tuning, Feature_extraction]')

parser.add_argument('--LwF', default=False,
                    help='[True, False] == Learning without Forgetting.')
parser.add_argument('--ld', default=0.02,
                    help='Importance rate for old tasks.')

parser.add_argument('--retro', default=False,
                    help='[True, False] == Progressive Distillation and Retrospection.')
parser.add_argument('--is_cycle', default=True,
                    help='[True, False] == Progressive Distillation and Retrospection with only one cycle.')

parser.add_argument('--encoder', default=False,
                    help='[True, False] == Encoder-based Lifelong Learning.')
parser.add_argument('--is_training_encoder', default=True,
                    help='[True, False] == [Training, Test] for a new encoder')
parser.add_argument('--enc_alpha', default=1e-2,
                    help='Importance rate for encoder.')
parser.add_argument('--enc_lr', default=1e-1,
                    help='Learning rate for encoder.')
parser.add_argument('--enc_num_epoch', default=80,
                    help='Number of epoch for encoder.')

parser.add_argument('--PNN', default=False,
                    help='Progressive Neural Networks.')

parser.add_argument('--EWC', default=False,
                    help='Elastic Weight Consolidation.')
parser.add_argument('--importance', default=2000,
                    help='Fisher scaling for Elastic Weight Consolidation.')

parser.add_argument('--memory', default=False,
                    help='[True, False] == Memory-based Continual Learning.')
parser.add_argument('--GB', default=False,
                    help='[True, False] == Growing a Brain V.S. Skip connections.')
parser.add_argument('--is_WWA', default=True,
                    help='[True, False] == WWA-CNN V.S. WA-CNN for both GB and GM.')
parser.add_argument('--is_scale', default=True,
                    help='[True, False] == Scaling parameter from ParseNet.')
parser.add_argument('--guided_learning', default=True,
                    help='[True, False] == Guided-Learning using zeta.')
parser.add_argument('--k_init', default=0.9,  # 0.2, 0.8, 0.9, 0.5, update it incrementally.
                    help='Initial K value for guided learning.')
parser.add_argument('--zeta', default=0.2,  # 2, 0.8, 0.2, 1
                    help='zeta for guided learning.')
parser.add_argument('--memory_test', default=False,
                    help='[True, False] == Only Memory_Acc V.S. Total_Acc')

parser.add_argument('--is_data_weight', default=False,
                    help='[True, False] == Data weight learning.')
parser.add_argument('--is_data_shift', default=False,
                    help='[True, False] == Data shift learning.')
parser.add_argument('--is_memory_distill', default=False,
                    help='Save memory outputs for conditionally shifted neurons.')
parser.add_argument('--is_KD_mem', default=False,
                    help='Knowledge distillation from the expert memory.')
parser.add_argument('--memory_shift', default=False,
                    help='Feature from the memory is shifted by additional parameters.')
args = parser.parse_args()


if not args.save:
    args.reuse = ''  # ['pre_alex_flowers_model', 'pre_alex_fe_flowers_birds_model', ...]
args = utils.bashRun(args)


##################
# Fix random seed.
random.seed(args.seed)
torch.manual_seed(args.seed)
use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
# torch.backends.cudnn.enabled = False  # cudnn library

################
# Load Datasets.
dataloaders, dataset_sizes = utils.load_data(args.dataset, args.batch_size, args.seed)
mini_dataloader, mini_dataset_size = [], []  # For retroModel.

###########################
# Define the network model.
model_name = 'models.' + args.cnn_type
model_args, train_args = [], []

if args.LwF:
    network_name = 'LwFModel'
    train_args = [args.ld]
elif args.retro:
    network_name = 'retroModel'
    mini_dataloader, mini_dataset_size = utils.load_mini_data(args.dataset[0:-1], args.batch_size, args.seed)
    train_args = [model_name, args.dataset[-1], args.num_classes[-1], mini_dataloader, mini_dataset_size, args.is_cycle, args.ld]
elif args.encoder:
    network_name = 'encModel'
    train_args = [args.dataset, args.ld, args.enc_alpha, args.enc_lr, args.enc_num_epoch,
                  args.weight_decay, args.is_training_encoder]
elif args.PNN:
    network_name = 'pnnModel'
elif args.EWC:
    network_name = 'ewcModel'
    model_args = [dataloaders, dataset_sizes]
    train_args = [args.importance]
elif args.memory:
    network_name = 'memoryModel_imagenet'  # WA-CNN or CNN-DM. (memoryModel / memoryModel_imagenet)
    model_args = [args.GB, args.is_WWA, args.is_scale, args.guided_learning, args.k_init, args.memory_test]
    train_args = [args.ld, args.zeta]
else:
    network_name = 'baseModel'  # FE or FT.


##################################
# Memory-based Continual Learning.
# --------------------------------
def main():
    model = eval(network_name + '.Model')(model_name, args.dataset, args.num_classes, *model_args,
                                          is_fine_tuning=args.is_fine_tuning)
    if use_gpu:
        model = model.cuda()

    # Define learning parameters.
    params = model.params(args.lr, is_fine_tuning=args.is_fine_tuning)

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Decay LR by a factor of gamma every step_size.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Load the model.
    start_epoch = 0
    model, start_epoch = utils.save_model(model, args.num_epoch, start_epoch, reuse=args.reuse, save_mode=False)

    # Train and Evaluate.
    if args.is_training:
        model = eval(network_name + '.train_model')(model, optimizer, exp_lr_scheduler, start_epoch, args.num_epoch,
                                                    dataloaders, dataset_sizes, *train_args)
        # Save the model.
        utils.save_model(model, args.num_epoch, start_epoch, reuse=args.reuse, save_mode=True)

        # Test the model after saving the model, because of out of memory error.
        if model.num_classifiers > 1:  # Continual Learning.
            print()
            for i in range(model.num_classifiers - 1):
                eval(network_name + '.test_model')(model, dataloaders, dataset_sizes, num_task=i)  # Test the model.
    else:
        # Test the model.
        for i in range(model.num_classifiers):
            eval(network_name + '.test_model')(model, dataloaders, dataset_sizes, num_task=i)


main()

print()
