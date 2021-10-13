import time
import torch
import torch.nn as nn
from model import utils
from model import LwFLoss
from torchvision import models
from torch.autograd import Variable


#######################################################
# Defining the Learning without Forgetting (LwF) model.
# -----------------------------------------------------
class Model(nn.Module):
    def __init__(self, model_name, dataset, num_classes, is_fine_tuning=True, pretrained=True,
                 network_name='LwFModel'):
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
                params = [{'params': self.features.parameters(), 'lr': 0.02 * lr},
                          {'params': self.fc6.parameters(), 'lr': 0.02 * lr},
                          {'params': self.fc7.parameters(), 'lr': 0.02 * lr}]

                for i in range(self.num_classifiers):
                    classifier_name = 'classifier' + str(i)
                    if i != self.num_classifiers - 1:
                        params = params + [{'params': getattr(self, classifier_name).parameters(), 'lr': 0.02 * lr}]
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


#####################
# Training the model.
def train_model(model, optimizer, scheduler, start_epoch, num_epochs, dataloaders, dataset_sizes, ld=0.02):

    # Define dataloader & dataset_size
    dataloader, dataset_size = dataloaders[model.num_classifiers-1], dataset_sizes[model.num_classifiers-1]

    # Define Criterion for loss.
    criterion = nn.CrossEntropyLoss()
    LwF_criterion = LwFLoss.LwFLoss()  # LwF_Loss

    # Gen_output for LwFLoss.
    prev_labels = {}
    prev_labels = utils.gen_output(model, dataloader, prev_labels)

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
                outputs = model(inputs)
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

        # if model.num_classifiers > 1:  # Continual Learning.
        #     if (epoch % 2 == 0 and epoch < 10) or (epoch % 10 == 0) or (epoch == num_epochs-1):
        #         test_model(model, dataloaders, dataset_sizes, num_task=0)  # Test the model.
        #     print()

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
        outputs = model(inputs)
        _, preds = torch.max(outputs[num_task].data, 1)  # To check Ac (Accuracy of total model).

        loss = criterion(outputs[num_task], labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size['test']
    epoch_acc = running_corrects / dataset_size['test']

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
