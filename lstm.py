import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, roc_curve, auc,f1_score

def runLSTM(name):
    #get pickle file
    basic = os.getcwd()
    data_path =  basic +"/data_" + name + "_x.pickle"
    label_path = basic + "/data_" + name + "_y.pickle"

    data = pickle.load(open(data_path, "rb"))
    label = pickle.load(open(label_path, "rb")).reshape(data.shape[0])
    
    # Hyper Parameters
    # Hyper Parameters
    EPOCH = 10             # train the training data n times
    BATCH_SIZE = 20        # set batch size
    HIDDEN_SIZE = 16       # set hidden out size
    LR = 0.01              # learning rate

    #split train and test data
    train_x, valid_x, train_y, valid_y = train_test_split(data, label, train_size=0.8, random_state=100)

    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    valid_x, valid_y = torch.from_numpy(valid_x), torch.from_numpy(valid_y)
    #set the train loader
    train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(train_x, train_y),
                                            batch_size=BATCH_SIZE, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #lstm class
    class BiRNN(nn.Module):
        #initial parameters
        def __init__(self, size,hidden_size=HIDDEN_SIZE, num_layers=1, num_classes=2):
            super(BiRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.size = size
            self.lstm = nn.LSTM(self.size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

        def forward(self, x):
            # Set initial states
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out


    # calculate feature importance class
    def feature_import(data, label, EPOCH,BATCH_SIZE,LR, accuracy):
        change_accu = np.zeros((0,0))

        #set new lstm which input size is 8
        model = BiRNN(8)
        for i in range(data.shape[2]):
            data_x = np.delete(data, i, axis=2)
            #split train and test data
            train_x, valid_x, train_y, valid_y = train_test_split(data_x, label, train_size=0.8, random_state=100)
            train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
            valid_x, valid_y = torch.from_numpy(valid_x), torch.from_numpy(valid_y)
            train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(train_x, train_y),batch_size=BATCH_SIZE, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all rnn parameters
            loss_func = nn.CrossEntropyLoss()

            for epoch in range(EPOCH):
                for step, (x, y) in enumerate(train_loader):  # gives batch data
                    b_x = Variable(x.float())  # batch x
                    b_y = Variable(y.long())  # batch y
                    output = model(b_x)  # rnn output
                    loss = loss_func(output, b_y)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    if step % 5 == 0:
                        valid_output = model(Variable(valid_x.float()))
                        train_output = model(Variable(train_x.float()))

                        pred_train = torch.max(train_output, 1)[1].data.numpy().squeeze()
                        pred_valid = torch.max(valid_output, 1)[1].data.numpy().squeeze()
                        train_accu = sum(pred_train == train_y.data.numpy()) / float(train_y.numpy().size)
                        valid_accu = sum(pred_valid == valid_y.data.numpy()) / float(valid_y.numpy().size)
                if epoch == EPOCH-1:
                    change = accuracy  - valid_accu - train_accu
                    change_accu = np.append(change_accu,[change])
        return change_accu

    #get lstm model
    rnn = BiRNN(9)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all rnn parameters
    loss_func = nn.CrossEntropyLoss()

    #to initial all the variables to store data
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
    f = plt.figure(figsize=(12, 5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    total_accu = np.zeros((0,0))
    total_prec = np.zeros((0,0))
    total_fscore = np.zeros((0,0))
    total_recall=np.zeros((0,0))
    total_roc_auc =np.zeros((0,0))


    for times in range(100):        # to run 100 times for getting average accuracy
        for epoch in range(EPOCH):
            for step, (x, y) in enumerate(train_loader):  # gives batch data
                b_x = Variable(x.float())  # batch x
                b_y = Variable(y.long())  # batch y
                output = rnn(b_x)  # rnn output
                loss = loss_func(output, b_y)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                if step % 5 == 0:
                    valid_output = rnn(Variable(valid_x.float()))
                    train_output = rnn(Variable(train_x.float()))
                    train_loss = loss_func(train_output, train_y.long())
                    valid_loss = loss_func(valid_output, valid_y.long())


                    pred_train = torch.max(train_output, 1)[1].data.numpy().squeeze()
                    pred_valid = torch.max(valid_output, 1)[1].data.numpy().squeeze()
                    train_accu = sum(pred_train == train_y.data.numpy()) / float(train_y.numpy().size)
                    valid_accu = sum(pred_valid == valid_y.data.numpy()) / float(valid_y.numpy().size)
                    if times == 1:
                        #get train and valid loss and accuracy
                        train_losses.append(train_loss.data)
                        valid_losses.append(valid_loss.data)
                        train_accuracy.append(train_accu)
                        valid_accuracy.append(valid_accu)

            #get the last epoch to calculate specific metrics
            if epoch == 9 and valid_accu > 0.5:
                accu = valid_accu
                total_accu=np.append(total_accu,[accu])
                total_recall = np.append(total_recall,[recall_score(pred_valid,valid_y)])
                total_fscore = np.append(total_fscore,[f1_score(pred_valid,valid_y)])
                total_prec = np.append(total_prec, [precision_score(pred_valid, valid_y)])
                try:
                    total_roc_auc = np.append(total_roc_auc, [roc_auc_score(pred_valid, valid_y)])
                except ValueError:
                    pass
                continue

    # calculate mean of all metrics
    mean_accu = np.mean(total_accu)
    mean_prec = np.mean(total_prec)
    mean_recall = np.mean(total_recall)
    mean_fscore = np.mean(total_fscore)
    mean_roc_auc = np.mean(total_roc_auc)
    # print metrics
    print('Accuracy:',mean_accu)
    print('Precision:',mean_prec)
    print('Recall:',mean_recall)
    print('f1score:',mean_fscore)
    print('roc and auc score:',mean_roc_auc)

    # draw loss and accuracy plot
    ax1.plot(train_losses, label="train loss")
    ax1.plot(valid_losses, label="valid loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')

    ax1.legend()
    ax2.plot(train_accuracy, label="train acc")
    ax2.plot(valid_accuracy, label="valid acc")
    ax2.legend()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.show()


    #calculate feature importance and draw the bar chart
    ac = feature_import(data,label,EPOCH,BATCH_SIZE,LR, accuracy = valid_accu + train_accu)
    print(ac)
    input_keys = ['Walk','Run', 'Noise', 'conversation_freq', 'conversation_time', 'dark_freq', 'dark_time','call_log_spark2','sms_spark']
    plt.barh(input_keys,ac)
    plt.xlabel('importance rate')
    plt.show()

