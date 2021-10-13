import time
import torch
import torch.nn as nn
from model import utils
from model import LwFLoss
from itertools import cycle, islice
from torchvision import models
from torch.autograd import Variable


################################################################
# Defining the Progressive Distillation and Retrospection model.
# --------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, model_name, dataset, num_classes, is_fine_tuning=True, pretrained=True,
                 network_name='retroModel'):
        super(Model, self).__init__()

        prev_model = eval(model_name)(pretrained=True)

        if not is_fine_tuning:  # Feature-extraction.
            for param in prev_model.parameters():
                param.requires_grad = False

        # Total number of classifiers.
        self.num_classifiers = len(num_classes)

        # Define the base model.
        self.features = prev_model.features
        self.fc6 = nn.Sequential(*list(prev_model.classifier.children())[:3])
        self.fc7 = nn.Sequential(*list(prev_model.classifier.children())[3:6])

        # self.classifier = nn.Linear(prev_model.classifier._modules['6'].in_features, num_classes).
        for i, num_class in enumerate(num_classes):
            classifier_name = 'classifier' + str(i)
            setattr(self, classifier_name, nn.Linear(prev_model.classifier._modules['6'].in_features, num_class))

            # If continual_learning & pretrained & before a new classifier, load the saved model.
            if (self.num_classifiers > 1) and pretrained and (i == self.num_classifiers - 2):
                if 'imagenet' in dataset[i]:
                    setattr(self, classifier_name, prev_model.classifier[6])
                else:
                    self.load_model(dataset[0:-1], model_name, network_name)

    # Load the saved model.
    def load_model(self, dataset, model_name, network_name):
        saved_model_name = network_name + '_'
        for data_name in dataset:
            saved_model_name = saved_model_name + data_name + '_'
        if 'vgg16' in model_name:  # vgg16 model.
            saved_model_name = saved_model_name + 'vgg'
        else:  # alexnet model.
            saved_model_name = saved_model_name + 'model'

        checkpoint = torch.load(saved_model_name)
        self.load_state_dict(checkpoint['state_dict'])  # Containing ['bias', 'weight'].

    # Define parameters to be trained.
    def params(self, lr, is_fine_tuning=True):
        if is_fine_tuning:
            if self.num_classifiers > 1:
                params = [{'params': self.features.parameters(), 'lr': 0.015 * lr},  # (0.01 or 0.015)
                          {'params': self.fc6.parameters(), 'lr': 0.015 * lr},
                          {'params': self.fc7.parameters(), 'lr': 0.015 * lr}]

                for i in range(self.num_classifiers):
                    classifier_name = 'classifier' + str(i)
                    if i != self.num_classifiers - 1:
                        params = params + [{'params': getattr(self, classifier_name).parameters(), 'lr': 0.015 * lr}]
                    else:
                        params = params + [{'params': getattr(self, classifier_name).parameters()}]
            else:
                params = self.parameters()
        else:  # Feature-Extraction.
            classifier_name = 'classifier' + str(self.num_classifiers - 1)
            params = [{'params': getattr(self, classifier_name).parameters()}]

        return params

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        fc6 = self.fc6(features)
        fc7 = self.fc7(fc6)

        outputs = []
        for i in range(self.num_classifiers):
            classifier_name = 'classifier' + str(i)
            output = getattr(self, classifier_name)(fc7)

            outputs = outputs + [output]

        return outputs


#######################################################
# Defining the Expert CNN model utilized in retroModel.
# -----------------------------------------------------
class ExpertModel(nn.Module):
    def __init__(self, model_name, dataset, num_class, network_name='retroModel'):
        super(ExpertModel, self).__init__()

        prev_model = eval(model_name)(pretrained=True)

        # Define the base model.
        self.features = prev_model.features
        self.fc6 = nn.Sequential(*list(prev_model.classifier.children())[:3])
        self.fc7 = nn.Sequential(*list(prev_model.classifier.children())[3:6])
        self.classifier0 = nn.Linear(prev_model.classifier._modules['6'].in_features, num_class)

        self.load_model(dataset, model_name, network_name)

    # Load the saved model.
    def load_model(self, dataset, model_name, network_name):
        if 'vgg16' in model_name:  # vgg16 model.
            saved_model_name = network_name + '_' + dataset + '_vgg'
        else:  # alexnet.
            saved_model_name = network_name + '_' + dataset + '_model'

        checkpoint = torch.load(saved_model_name)
        self.load_state_dict(checkpoint['state_dict'])  # Containing ['bias', 'weight'].

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        fc6 = self.fc6(features)
        fc7 = self.fc7(fc6)

        outputs = [self.classifier0(fc7)]

        return outputs


def train_dataloader(num_classifiers, dataloader, dataset_sizes, mini_dataloaders, mini_dataset_sizes, phase='train'):

    # Find max_dataset_size and index.
    total_dataset_size = []
    for i in range(num_classifiers - 1):
        total_dataset_size = total_dataset_size + [mini_dataset_sizes[i]['mini']]
    total_dataset_size = total_dataset_size + [dataset_sizes[num_classifiers - 1]['train']]

    max_dataset_size = max(total_dataset_size)
    max_dataset_size_index = total_dataset_size.index(max(total_dataset_size))

    # Define fixed_cycle function.
    fixed_cycle = lambda lst, n: islice(cycle(lst), n)

    # Get total_dataloader from ordered_dataloaders.
    ordered_dataloaders = []
    for i in range(num_classifiers):
        if i < (num_classifiers - 1):
            if i != max_dataset_size_index:
                ordered_dataloaders = ordered_dataloaders + [fixed_cycle(mini_dataloaders[i]['mini'], max_dataset_size)]
            else:
                ordered_dataloaders = ordered_dataloaders + [mini_dataloaders[i]['mini']]
        else:
            if i != max_dataset_size_index:
                ordered_dataloaders = ordered_dataloaders + [fixed_cycle(dataloader[phase], max_dataset_size)]
            else:
                ordered_dataloaders = ordered_dataloaders + [dataloader[phase]]

    total_dataloader = zip(*ordered_dataloaders)

    return total_dataloader, max_dataset_size


#####################
# Training the model.
def train_model(model, optimizer, scheduler, start_epoch, num_epochs, dataloaders, dataset_sizes,
                model_name, dataset, num_class, mini_dataloaders, mini_dataset_sizes, is_cycle=True, ld=0.02):

    # Define dataloader & dataset_size
    dataloader, dataset_size = dataloaders[model.num_classifiers-1], dataset_sizes[model.num_classifiers-1]

    # Define Criterion for loss.
    criterion = nn.CrossEntropyLoss()
    LwF_criterion = LwFLoss.LwFLoss(tau=1)  # LwF_Loss
    retro_criterion = LwFLoss.LwFLoss(tau=2)  # LwF_Loss

    # Define the expert CNN.
    expert_model = ExpertModel(model_name, dataset, num_class)
    if torch.cuda.is_available():
        expert_model = expert_model.cuda()

    # Gen_output for LwFLoss.
    prev_labels, curr_labels = {}, {}
    for i in range(len(mini_dataloaders)):
        prev_labels = utils.gen_output(model, mini_dataloaders[i], prev_labels, retro=True, mini_data=True)  # Retrospection.

    curr_labels = utils.gen_output(expert_model, dataloader, curr_labels, retro=True)  # Distillation.

    # Delete the expert CNN to free GPU memory.
    classifier_name = 'classifier' + str(model.num_classifiers - 1)
    setattr(model, classifier_name, expert_model.classifier0)  # Transfer the classifier of the expert CNN.
    del expert_model

    best_model_wts = model.state_dict()
    torch.save({'model': best_model_wts}, 'curr_best_model_wts')
    best_loss = 0.0
    best_acc, best_mean_acc, best_curr_acc = 0.0, 0.0, 0.0

    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(start_epoch + epoch, start_epoch + num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        max_dataset_size, accumulated_batch = 0, [0] * model.num_classifiers
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode

                # Find max_dataset_size and index.
                total_dataloader, max_dataset_size = train_dataloader(model.num_classifiers, dataloader, dataset_sizes, mini_dataloaders, mini_dataset_sizes)

            else:
                model.train(False)  # Set model to evaluate mode
                total_dataloader = dataloader[phase]

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, data in enumerate(total_dataloader):
                # get the inputs
                if phase == 'train':
                    mini_inputs, mini_labels, mini_addresses = [], [], []
                    for k in range(model.num_classifiers-1):
                        mini_input, mini_label, mini_address = data[k]

                        mini_inputs = mini_inputs + [mini_input]
                        mini_labels = mini_labels + [mini_label]
                        mini_addresses = mini_addresses + [mini_address]

                    inputs, labels, addresses = data[-1]
                else:
                    inputs, labels, _ = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    if phase == 'train':
                        for k in range(model.num_classifiers - 1):
                            mini_inputs[k], mini_labels[k] = Variable(mini_inputs[k].cuda()), Variable(mini_labels[k].cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'train':
                    outputs = model(torch.cat((inputs, *mini_inputs), dim=0))

                    split_size, mini_outputs = 0, []
                    for k in range(model.num_classifiers - 1):
                        _, mini_output = utils.split_tensor(outputs[k], len(inputs) + split_size)  # Retrospection.

                        mini_outputs = mini_outputs + [mini_output[:len(mini_inputs[k])]]
                        split_size = split_size + len(mini_inputs[k])

                    curr_outputs, _ = utils.split_tensor(outputs[-1], len(inputs))  # Distillation.

                    _, preds = torch.max(curr_outputs.data, 1)  # You can use "topk" function.

                    # Calculate distillation loss.
                    distillation_loss = 0
                    if is_cycle:
                        # wrap curr_labels in Variable for out of memory.
                        curr_labels_i = []
                        for address in addresses:
                            curr_labels_i.append(curr_labels[address][0])  # from expert_model.

                        if torch.cuda.is_available():
                            curr_labels_i = Variable(torch.stack(curr_labels_i).cuda())

                        distillation_loss = LwF_criterion(curr_outputs, curr_labels_i)
                    else:
                        # Identify whether the current batch is in a cycle or not.
                        if accumulated_batch[-1] < dataset_size['train']:
                            accumulated_batch[-1] = accumulated_batch[-1] + len(inputs)

                            # wrap curr_labels in Variable for out of memory.
                            curr_labels_i = []
                            for address in addresses:
                                curr_labels_i.append(curr_labels[address][0])  # from expert_model.

                            if torch.cuda.is_available():
                                curr_labels_i = Variable(torch.stack(curr_labels_i).cuda())

                            distillation_loss = LwF_criterion(curr_outputs, curr_labels_i)

                    # Calculate retrospection loss.
                    retrospection_loss = 0
                    for k in range(model.num_classifiers - 1):
                        if is_cycle:
                            # wrap prev_labels in Variable for out of memory.
                            prev_labels_i = []
                            for address in mini_addresses[k]:
                                prev_labels_i.append(prev_labels[address][k])  # k-th classifier labels.

                            if torch.cuda.is_available():
                                prev_labels_i = Variable(torch.stack(prev_labels_i).cuda())

                            retrospection_loss = retrospection_loss + retro_criterion(mini_outputs[k], prev_labels_i)
                        else:
                            # Identify whether the current batch is in a cycle or not.
                            if accumulated_batch[k] < mini_dataset_sizes[k]['mini']:
                                accumulated_batch[k] = accumulated_batch[k] + len(mini_inputs[k])

                                # wrap prev_labels in Variable for out of memory.
                                prev_labels_i = []
                                for address in mini_addresses[k]:
                                    prev_labels_i.append(prev_labels[address][k])  # k-th classifier labels.

                                if torch.cuda.is_available():
                                    prev_labels_i = Variable(torch.stack(prev_labels_i).cuda())

                                retrospection_loss = retrospection_loss + retro_criterion(mini_outputs[k], prev_labels_i)

                    # Two Knowledge Distillation Loss.
                    r_k = 2 if is_cycle else 1
                    loss = distillation_loss + r_k * retrospection_loss
                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[-1].data, 1)  # You can use "topk" function.

                    loss = criterion(outputs[-1], labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == 'train':
                epoch_loss = running_loss / max_dataset_size
                epoch_acc = running_corrects / max_dataset_size
            else:
                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc

        num_list = [0.525, 0.569, 0.8285]
        if model.num_classifiers > 1:  # Continual Learning.
            if (epoch_acc > num_list[model.num_classifiers-2]) or (epoch_acc > best_acc - 0.003):
                prev_epoch_acc = epoch_acc
                for j in range(model.num_classifiers-1):
                    prev_epoch_acc = prev_epoch_acc + test_model(model, dataloaders, dataset_sizes, num_task=j)  # Test the model.
                prev_epoch_acc = prev_epoch_acc / (model.num_classifiers)

                # deep copy the model
                if prev_epoch_acc > best_mean_acc:
                    best_mean_acc = prev_epoch_acc
                    best_curr_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save({'model': best_model_wts}, 'curr_best_model_wts')
                    print('Mean Acc: {:.4f}'.format(best_mean_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Loss: {:4f} Acc: {:4f}'.format(best_loss, best_acc))
    print('Best curr Loss: {:4f} Acc: {:4f}'.format(best_loss, best_curr_acc))

    # load the best model.
    checkpoint = torch.load('curr_best_model_wts')
    model.load_state_dict(checkpoint['model'])

    return model


#################
# Test the model.
def test_model(model, dataloaders, dataset_sizes, num_task):

    # Define dataloader & dataset_size
    dataloader, dataset_size = dataloaders[num_task], dataset_sizes[num_task]

    # Define Criterion for loss.
    criterion = nn.CrossEntropyLoss()

    model.train(False)

    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(dataloader['test']):
        inputs, labels, _ = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs[num_task].data, 1)  # To check Ac (Accuracy of total model).

        loss = criterion(outputs[num_task], labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size['test']
    epoch_acc = running_corrects / dataset_size['test']

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return epoch_acc
