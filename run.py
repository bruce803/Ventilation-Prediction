import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from function import StepFunction, concordance_index_censored
from nonparametric import compute_counts
from datapreparation import data_preparation
try:
    import cPickle as pickle
except:
    import pickle


#####################
# Build our model
#####################

import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
# DDD
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        #         Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)


    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        lstm_out, self.hidden = self.lstm(inputs)  # [715, 200, hidden_dim]
        #         lstm_out, self.hidden = self.lstm(input.view(715, self.batch_size, -1)) #[715, 200, hidden_dim]
        # print('the shape of linear output', lstm_out[-1].size()) #([100, 20])

        likelihood = self.linear(lstm_out[-1])
        # print('the likelihood before sum:', likelihood.sum(-1)) #([100, 1])

        #         y_exp = torch.exp(lstm_out)
        #         print('the shape of y_exp', y_exp.size())

        return likelihood.sum(-1)


class LSTM_ReLu(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM_ReLu, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        #         Define the output layer
        self.linear = nn.Linear(self.hidden_dim, 128)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, output_dim)


    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        lstm_out, self.hidden = self.lstm(inputs)  # [715, 200, hidden_dim]
        #         lstm_out, self.hidden = self.lstm(input.view(715, self.batch_size, -1)) #[715, 200, hidden_dim]
        # print('the shape of linear output', lstm_out[-1].size()) #([100, 20])


        likelihood = self.linear(lstm_out[-1])
        likelihood = self.relu(likelihood)
        likelihood = self.linear1(likelihood)
        # print('the likelihood before sum:', likelihood.sum(-1)) #([100, 1])

        #         y_exp = torch.exp(lstm_out)
        #         print('the shape of y_exp', y_exp.size())

        return likelihood.sum(-1)


class LSTM_MASK(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM_MASK, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        # self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)

        #         Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)


    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        hidden = self.init_hidden(inputs.size(1))
        seq_len = inputs.size(0)
        seq_num = inputs.size(1)
        mask = torch.tensor(inputs[...,-1]>0, dtype=torch.float32).cuda()
        # print(mask.size())
        # mask1 = torch.ones([seq_len, seq_num], dtype=torch.float32).cuda()

        # mask:size:seq_len x seq_num 715x200
        for i in range(seq_len):
            mask_i = mask[i,:]
            inputs_i = inputs[i,:,:]
            inputs_i = inputs_i.view(1,seq_num,-1)
            lstm_out, hidden = self.lstm(inputs_i,hidden)  # [715, 200, hidden_dim]
            hidden_ = mask_i[None,:,None]*hidden[0]
            hidden_c = mask_i[None,:,None]*hidden[1]
            hidden = [hidden_,hidden_c]

        #         lstm_out, self.hidden = self.lstm(input.view(715, self.batch_size, -1)) #[715, 200, hidden_dim]
        # print('the shape of linear output', lstm_out[-1].size()) #([100, 20])

        likelihood = self.linear(lstm_out[-1])
        # print('the likelihood before sum:', likelihood.sum(-1)) #([100, 1])

        #         y_exp = torch.exp(lstm_out)
        #         print('the shape of y_exp', y_exp.size())
        # print('the likelihood before sum:', likelihood.sum(-1))

        return likelihood.sum(-1)


class NLL_loss(torch.nn.Module):
    def __init__(self):
        super(NLL_loss, self).__init__()

    def forward(self, risk, event_indicator):
        '''
        risk: sequence output of lstm,with shape (num_samples,)
        event_indicator: (num_samples,)
        '''

        # sort data by the observed time such that higher row number indicates higher survival time
        #         ind = np.argsort(-event_time, kind="mergesort")
        #         log_risk = torch.log(torch.cumsum(risk))
        #         uncensored_likelihood = risk

        hazard_ratio = torch.exp(risk)  # risk: lstm_out,
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * event_indicator
        n_samples = event_indicator.sum()

        return -censored_likelihood.sum() / n_samples


class NLL_loss_v1(torch.nn.Module):
    def __init__(self):
        super(NLL_loss_v1, self).__init__()

    def forward(self, risk, event_indicator, time):
        '''
        risk: sequence output of lstm,with shape (num_samples,)
        event_indicator: (num_samples,)
        time: (num_samples,)
        '''
        num_samples = time.shape[0]
        loss = k = 0
        lis = []
        for i in range(num_samples):
            risk_set = 0
            t_i = time[i]  # 1 ,2
            while k < num_samples and t_i == time[k]:
                risk_set += torch.exp(risk[k])
                k = k + 1
            lis.append(risk_set)
        risk_sum = torch.FloatTensor(lis)
        risk_sum = risk_sum.cuda()
        #         print(risk_sum.size())
        risk_cumsum = torch.cumsum(risk_sum, dim=0)

        risk_minus = risk - torch.log(risk_cumsum)
        loss = -(risk_minus * event_indicator).sum() / num_samples
        return loss.cuda()


class Dataset_mimic(Dataset):

    def __init__(self, pklFilePath, transform=None):
        with open(pklFilePath, 'rb') as fp:
            self.data = pickle.load(fp)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        covariates = self.data[index][0]
        indicator = self.data[index][1]
        occurrence_time = self.data[index][2]
        return np.float32(covariates), np.float32(indicator), np.float32(occurrence_time)

class Dataset_mimic_lis(Dataset):

    def __init__(self, lis, transform=None):
        self.data = lis

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        covariates = self.data[index][0]
        indicator = self.data[index][1]
        occurrence_time = self.data[index][2]
        return np.float32(covariates), np.float32(indicator), np.float32(occurrence_time)

def extract_data_from_pkl(pklfile):
    with open(pklfile, 'rb') as f:
        test_sample_list = pickle.load(f)

    return [test_sample_list[i][0] for i in range(len(test_sample_list))], \
           [test_sample_list[i][1] for i in range(len(test_sample_list))], \
           [test_sample_list[i][2] for i in range(len(test_sample_list))]


def extract_data_from_lis(lis):

    return [lis[i][0] for i in range(len(lis))], \
           [lis[i][1] for i in range(len(lis))], \
           [lis[i][2] for i in range(len(lis))]


def fit_baseline_hazard_function(risk_score, event, time):

    order = np.argsort(time, kind="mergesort")
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk = compute_counts(event, time, order)

    divisor = np.empty(n_at_risk.shape, dtype=np.float_)
    value = np.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    y = np.cumsum(n_events / divisor)
    return StepFunction(uniq_times, y)


def predict_survival_function(risk_score, event, time):

    cum_baseline_hazard = fit_baseline_hazard_function(risk_score, event, time)
    baseline_survival_ = StepFunction(cum_baseline_hazard.x, np.exp(-cum_baseline_hazard.y))

    n_samples = risk_score.shape[0]
    funcs = np.empty(n_samples, dtype=np.object_)
    for i in range(n_samples):
        funcs[i] = StepFunction(x=baseline_survival_.x,
                                y=np.power(baseline_survival_.y, risk_score[i]))
    return funcs
#####################
# Set parameters
#####################

# Network params
input_size = 22
# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = False
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers
h1 = 1024
# output_dim = 1
num_layers = 1
learning_rate = 1e-8
num_epochs = 5
dtype = torch.float

'''load data'''
print('###### Begin to load data...')
mimic = 'mimic.csv'
mimic3 = pd.read_csv(mimic)
sample_list = data_preparation(mimic3, 50)

train = Dataset_mimic_lis(sample_list[0:20000], None)
test = Dataset_mimic_lis(sample_list[20000:25052], None)

trainloader = DataLoader(train, batch_size=200, shuffle=True, num_workers=6)
testloader = DataLoader(test, batch_size=100, shuffle=True, num_workers=2)

###load model
model = LSTM_ReLu(lstm_input_size, h1, num_layers=num_layers)
criterion = NLL_loss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.cuda()

###training
hist = np.zeros(num_epochs)
for t in range(num_epochs):
    model.train()
    for ind_batch, local_batch in enumerate(trainloader):

        batch_x, batch_indicator, batch_time = local_batch

        sort_inx = np.argsort(-batch_time, kind='mergesort')
        batch_x = batch_x[sort_inx]  # currently, the 1-st dim of batch_x is still batch_size

        batch_indicator = batch_indicator[sort_inx]
        batch_time = batch_time[sort_inx]

        batch_x = batch_x.permute(1, 0, 2)  # now, the 1-st dim of batch_x is 715
        #         print(batch_x.size())
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden(batch_x.size(1))  # input batch size

        # input data
        batch_indicator = batch_indicator.cuda()
        batch_x = batch_x.cuda()
        batch_time = batch_time.cuda()

        # Forward pass
        outputs = model(batch_x)
        #         print(outputs)
        #         print('shape of lstm_out:', outputs.shape)

        loss = criterion(outputs, batch_indicator)

        if t % 10 == 0:
            print("Epoch ", t, "loss: ", loss.item())
        #             print("Epoch ", t, "loss permuted: ", loss1.item())

        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()


## testing

model.eval()
riskScore = []
# test_X = []
# event = []
with torch.no_grad():
    for inx_batch, batch in enumerate(testloader):
        #         if inx_batch > 24:
        #             break

        test_x, test_indicator, test_time = batch
        #         print(test_x.size())
        #         test_X.append(test_x)

        test_x = test_x.cuda()
        #         test_indicator = test_indicator.cuda()
        #         test_time = test_time.cuda()

        test_x = test_x.permute(1, 0, 2)  # align dim with model(num_cell, batch_size, hidden_dim)
        #         print(test_x.size())
        risk_score = model(test_x)
        #         print(risk_score.shape)
        riskScore.append(risk_score)

rs_tensor = torch.cat(riskScore, dim=0)
rs_array = rs_tensor.cpu().numpy()
# print('rsarray shape:', rs_array.shape)


##estimate survival function
# X_list, event_list, time_list = extract_data_from_pkl('train-2000.pkl') #load testing data
X_list, event_list, time_list = extract_data_from_lis(sample_list[20000:25052]) #load testing data
surFun = predict_survival_function(rs_array, np.asarray(event_list), np.asarray(time_list))
# print('len of event list',len(event_list))
# print('len of time_list', len(time_list))
# print('len of X_list', len(X_list))
# print(surFun)

'''
figure(num=None, figsize=(5, 3), dpi=100, facecolor='w', edgecolor='k')

plt.step(surFun[8].x,surFun[8].y, linewidth=2)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()
# plt.savefig('sf1.pdf',dpi=200, bbox_inches='tight')
'''

#caculating C-index
c_index = concordance_index_censored(np.asarray(event_list,dtype=bool), np.asarray(time_list), rs_array)
print('the C-index value is:', c_index[0])