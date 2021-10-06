import os
import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from model import fixed_transforms, fixed_dataloader

plt.ion()  # interactive mode: pyplot functions automatically draw to the screen.


########################
# Save & Load the model.
def save_model(model, num_epoch, start_epoch, reuse, save_mode=False):
    if save_mode:   # save the model.
        if reuse:
            print("=> saving checkpoint '{}'".format(reuse))
            torch.save({
                'epoch': (num_epoch + start_epoch),
                'state_dict': model.state_dict(),
            }, reuse)
    else:           # load the model.
        if reuse:
            if os.path.isfile(reuse):
                print("=> loading checkpoint '{}'...".format(reuse))
                checkpoint = torch.load(reuse)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])  # Containing ['bias', 'weight'].
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(reuse, checkpoint['epoch']))
            else:
                # pass
                print("=> no checkpoint found at '{}'".format(reuse))

        return model, start_epoch


################
# Load datasets.
def load_data(datasets, batch_size, seed):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': fixed_transforms.Compose([  # fixed_
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.RandomCrop(224),  # 256x256 image -> 224x224 image
            # transforms.RandomHorizontalFlip(),
            fixed_transforms.RandomCrop(224),  # fixed_
            fixed_transforms.RandomHorizontalFlip(),  # fixed_
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': fixed_transforms.Compose([  # fixed_
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # (fixed_dataloader.ImageFolder), (datasets.ImageFolder)
    dataloaders, dataset_sizes = {}, {}
    for i, dataset in enumerate(datasets):
        image_datasets = {x: fixed_dataloader.ImageFolder(os.path.join(('./dataset/' + dataset), x), data_transforms[x])
                          for x in ['train', 'test']}
        dataloaders[i] = {x: fixed_dataloader.DataLoader(image_datasets[x], batch_size=batch_size,  # torch.utils.data.DataLoader
                                                         shuffle=True, num_workers=8, pin_memory=True, seed=seed)
                          for x in ['train', 'test']}
        dataset_sizes[i] = {x: len(image_datasets[x]) for x in ['train', 'test']}
        # class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes


####################
# Load mini dataset.
def load_mini_data(datasets, batch_size, seed):
    # Data augmentation and normalization for training
    data_transforms = {
        'mini': fixed_transforms.Compose([  # fixed_
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.RandomCrop(224),  # 256x256 image -> 224x224 image
            # transforms.RandomHorizontalFlip(),
            fixed_transforms.RandomCrop(224),  # fixed_
            fixed_transforms.RandomHorizontalFlip(),  # fixed_
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # (fixed_dataloader.ImageFolder), (datasets.ImageFolder)
    dataloaders, dataset_sizes = {}, {}
    for i, dataset in enumerate(datasets):
        image_dataset = {x: fixed_dataloader.ImageFolder(os.path.join(('./dataset/' + dataset), x), data_transforms[x])
                         for x in ['mini']}
        dataloaders[i] = {x: fixed_dataloader.DataLoader(image_dataset[x], batch_size=batch_size,  # torch.utils.data.DataLoader
                                                     shuffle=True, num_workers=8, pin_memory=True, seed=seed)
                      for x in ['mini']}
        dataset_sizes[i] = {x: len(image_dataset[x]) for x in ['mini']}

    return dataloaders, dataset_sizes


##############################################
# Generate the responses of the original model
def gen_output(model, dataloader, prev_labels, feature_model=None, retro=False, mini_data=False, network_name='baseModel'):
    model.train(False)
    if feature_model is not None:
        feature_model.train(False)

    if retro:  # For retroModel.
        if mini_data:
            folder = 'mini'
        else:
            folder = 'train'
    else:
        folder = 'train'

    for i, data in enumerate(dataloader[folder]):
        inputs, labels, addresses = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        if feature_model is not None:
            features = feature_model(inputs)
            features = features.view(features.size(0), 256 * 6 * 6)

            prev_outputs, _ = model(features)  # Encoder model.

            prev_labels = prev_labels + [prev_outputs.data.cpu()]
        else:
            if 'memoryModel' in network_name:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            if retro:  # For [retroModel].
                for k, _ in enumerate(outputs):
                    outputs[k] = outputs[k].data.cpu()

                zipped_outputs = list(zip(*outputs))

                for k, address in enumerate(addresses):
                    if not (address in prev_labels):
                        prev_labels[address] = []

                    prev_labels[address] = prev_labels[address] + list(zipped_outputs[k])
            else:  # For [LwF, Ours].
                for k, output in enumerate(outputs[0:-1]):
                    if not (k in prev_labels):
                        prev_labels[k] = []

                    prev_labels[k] = prev_labels[k] + [output.data.cpu()]

    return prev_labels


################################
# Compare weights of two models.
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly!')


#################################################
# Count the number of trainable model parameters.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


###################################
# Split a tensor into half tensors.
def split_tensor(a_tensor, length):
    # half = len(a_tensor) // 2
    # return a_tensor[:half], a_tensor[half:]
    return a_tensor[:length], a_tensor[length:]


###################################################################################
# Let's visualize a few training images so as to understand the data augmentations.
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


####################################
# Visualizing the model predictions.
def visualize_model(model, use_gpu, num_images, dataloders, class_names):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloders['test']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        prev_outputs, outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                plt.ioff()
                plt.show()

                return

    # Example code of visualize_model.

    # Get a batch(=batch_size) of training data
    # inputs, classes = next(iter(dataloders['train']))

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # utils.imshow(out, title=[class_names[x] for x in classes])

    # utils.visualize_model(model, use_gpu, args.batch_size, dataloders, class_names)


#######################
# Bash Running Setting.
def bashRun(args):
    # For bash running.
    if type(args.lr) is str:
        args.lr = float(args.lr)

    if type(args.seed) is str:
        args.seed = int(args.seed)

    if type(args.batch_size) is str:
        args.batch_size = int(args.batch_size)

    if type(args.weight_decay) is str:
        args.weight_decay = int(args.weight_decay)

    if str(args.memory) in ("True"):
        args.memory = True
    else:
        args.memory = False

    if str(args.GB) in ("True"):
        args.GB = True
    else:
        args.GB = False

    if str(args.is_WWA) in ("True"):
        args.is_WWA = True
    else:
        args.is_WWA = False

    if type(args.k_init) is str:
        args.k_init = float(args.k_init)

    if type(args.zeta) is str:
        args.zeta = float(args.zeta)

    if args.dataset is 'scenes':
        ld = 0.025
    else:
        ld = 0.02

    if type(args.ld) is str:
        args.ld = float(args.ld)

    return args
