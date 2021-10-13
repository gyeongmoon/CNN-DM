import time
import torch
import torch.nn as nn
from model import utils
from model import LwFLoss
from torchvision import models
from torch.autograd import Variable


############################################################
# Defining the CNN with Developmental Memory (CNN-DM) model.
# ----------------------------------------------------------
class Model(nn.Module):
    def __init__(self, model_name, dataset, num_classes, GB=False, is_WWA=True, is_scale=True, guided_learning=True,
                 k_init=0.5, memory_test=False, is_fine_tuning=True, pretrained=True, network_name='memoryModel'):
        super(Model, self).__init__()

        # Set options.
        self.GB = GB
        self.is_WWA = is_WWA
        self.is_scale = is_scale
        self.guided_learning = guided_learning
        self.memory_test = memory_test

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
            setattr(self, classifier_name, nn.Linear(prev_model.classifier._modules['6'].in_features + 2048, num_class))

            augClassifier_name = 'augClassifier' + str(i)
            if GB:  # WWA-CNN from "Growing a Brain" paper.
                if is_WWA:
                    setattr(self, augClassifier_name + '_0', nn.Sequential(
                        nn.Linear(256 * 6 * 6, 1024),
                        nn.ReLU(inplace=True),
                    ))
                    setattr(self, augClassifier_name + '_1', nn.Sequential(
                        nn.Linear(4096 + 1024, 2048),
                        nn.ReLU(inplace=True),
                    ))
                else:  # WA-CNN
                    setattr(self, augClassifier_name, nn.Sequential(
                        nn.Linear(4096, 2048),
                        nn.ReLU(inplace=True),
                    ))
            else:
                if is_WWA:  # Both features and fc6.
                    if 'vgg16' in model_name:
                        setattr(self, augClassifier_name, nn.Linear(512 * 7 * 7 + 4096, 2048))
                    else:
                        setattr(self, augClassifier_name, nn.Linear(256 * 6 * 6 + 4096, 2048))
                else:  # Only fc6.
                    setattr(self, augClassifier_name, nn.Linear(4096, 2048))

            if is_scale:
                scale = torch.randn(4096 + 2048).fill_(20)  # scale = [h^k, h^k+]
                if guided_learning:
                    k = k_init
                    scale[:4096] = scale[:4096] * k
                setattr(self, 'scale' + str(i), nn.Parameter(scale.cuda(), requires_grad=True))

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
                if self.GB:  # From here.
                    params = [{'params': self.features.parameters(), 'lr': 0.01 * lr},
                              {'params': self.fc6.parameters(), 'lr': 0.01 * lr},
                              {'params': self.fc7.parameters(), 'lr': 0.01 * lr}]

                    for i in range(self.num_classifiers):
                        if i != self.num_classifiers - 1:
                            params = params + [{'params': getattr(self, 'classifier' + str(i)).parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'augClassifier' + str(i) + '_0').parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'augClassifier' + str(i) + '_1').parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'scale' + str(i)), 'lr': 0.001 * lr}]
                        else:
                            params = params + [{'params': getattr(self, 'classifier' + str(i)).parameters()},
                                               {'params': getattr(self, 'augClassifier' + str(i) + '_0').parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'augClassifier' + str(i) + '_1').parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'scale' + str(i))}]
                else:
                    params = [{'params': self.features.parameters(), 'lr': 0.01 * lr},
                              {'params': self.fc6.parameters(), 'lr': 0.01 * lr},
                              {'params': self.fc7.parameters(), 'lr': 0.05 * lr}]

                    for i in range(self.num_classifiers):
                        if i != self.num_classifiers - 1:
                            params = params + [{'params': getattr(self, 'classifier' + str(i)).parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'augClassifier' + str(i)).parameters(), 'lr': 0.001 * lr},
                                               {'params': getattr(self, 'scale' + str(i)), 'lr': 0.001 * lr}]
                        else:
                            params = params + [{'params': getattr(self, 'classifier' + str(i)).parameters()},
                                               {'params': getattr(self, 'augClassifier' + str(i)).parameters()},
                                               {'params': getattr(self, 'scale' + str(i))}]
            else:
                # To train the memory for imagenet..
                # if (self.num_classifiers == 1) and 'imagenet' in dataset[0]:
                #     params = [{'params': getattr(self, 'classifier' + str(0)).parameters()},
                #               {'params': getattr(self, 'augClassifier' + str(0)).parameters()},
                #               {'params': getattr(self, 'scale' + str(0))}]

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

        outputs, augOutputs = [], []
        for i in range(self.num_classifiers):
            if self.GB:
                if self.is_WWA:  # Both features and fc6.
                    fc6_plus = getattr(self, 'augClassifier' + str(i) + '_0')(features)
                    fc7_plus = getattr(self, 'augClassifier' + str(i) + '_1')(torch.cat((fc6, fc6_plus), 1))
                else:  # Only fc6.
                    fc7_plus = getattr(self, 'augClassifier' + str(i))(fc6)
            else:
                if self.is_WWA:  # Both features and fc6.
                    fc7_plus = getattr(self, 'augClassifier' + str(i))(torch.cat((features, fc6), 1))
                else:  # Only fc6.
                    fc7_plus = getattr(self, 'augClassifier' + str(i))(fc6)  # Only fc6.

            classifier_name = 'classifier' + str(i)
            if self.is_scale:
                # ParseNet Normalization.
                norm_fc7 = fc7.div(torch.norm(fc7, 2, 1, keepdim=True).expand_as(fc7))
                norm_fc7_plus = fc7_plus.div(torch.norm(fc7_plus, 2, 1, keepdim=True).expand_as(fc7_plus))

                output = getattr(self, classifier_name)(torch.cat(
                    (getattr(self, 'scale' + str(i))[:4096].expand_as(norm_fc7) * norm_fc7,
                     getattr(self, 'scale' + str(i))[4096:].expand_as(norm_fc7_plus) * norm_fc7_plus), 1))

                zero_inputs = Variable(torch.zeros(norm_fc7.size()).cuda(), requires_grad=False)
                augOutput = getattr(self, classifier_name)(torch.cat(
                    (zero_inputs, getattr(self, 'scale' + str(i))[4096:].expand_as(norm_fc7_plus) * norm_fc7_plus), 1))

                augOutputs = augOutputs + [augOutput]
            else:
                output = getattr(self, classifier_name)(torch.cat((fc7, fc7_plus), 1))

            outputs = outputs + [output]

        return outputs, augOutputs


#####################
# Training the model.
def train_model(model, optimizer, scheduler, start_epoch, num_epochs, dataloaders, dataset_sizes, ld=0.02, zeta=1):

    # Define dataloader & dataset_size
    dataloader, dataset_size = dataloaders[model.num_classifiers-1], dataset_sizes[model.num_classifiers-1]

    # Define Criterion for loss.
    criterion = nn.CrossEntropyLoss()
    LwF_criterion = LwFLoss.LwFLoss()  # LwF_Loss

    # Gen_output for LwFLoss.
    prev_labels = {}
    if model.num_classifiers > 1:
        prev_labels = utils.gen_output(model, dataloader, prev_labels, network_name='memoryModel')

    best_model_wts = model.state_dict()
    torch.save({'model': best_model_wts}, 'curr_best_model_wts')
    best_loss = 0.0
    best_acc = 0.0

    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(start_epoch + epoch, start_epoch + num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, data in enumerate(dataloader[phase]):
                # get the inputs
                inputs, labels, _ = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, augOutputs = model(inputs)
                _, preds = torch.max(outputs[-1].data, 1)  # You can use "topk" function.

                if phase == 'train':
                    LwF_Loss = 0
                    for k in range(model.num_classifiers - 1):
                        # wrap prev_labels in Variable for out of memory.
                        if torch.cuda.is_available():
                            prev_labels_i = Variable(prev_labels[k][i].cuda())
                        else:
                            prev_labels_i = prev_labels[k][i]

                        LwF_Loss = LwF_Loss + LwF_criterion(outputs[k], prev_labels_i)

                    # CrossEntropyLoss + Knowledge Distillation Loss.
                    if model.guided_learning:
                        loss = criterion(outputs[-1], labels) + zeta * criterion(augOutputs[-1], labels) + ld * LwF_Loss
                    else:
                        loss = criterion(outputs[-1], labels) + ld * LwF_Loss
                else:
                    loss = criterion(outputs[-1], labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save({'model': best_model_wts}, 'curr_best_model_wts')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Loss: {:4f} Acc: {:4f}'.format(best_loss, best_acc))  # mems

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
        outputs, augOutputs = model(inputs)
        if model.memory_test:
            output = augOutputs[num_task]
        else:
            output = outputs[num_task]

        _, preds = torch.max(output.data, 1)  # To check Ac (Accuracy of model).

        loss = criterion(output, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size['test']
    epoch_acc = running_corrects / dataset_size['test']

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
