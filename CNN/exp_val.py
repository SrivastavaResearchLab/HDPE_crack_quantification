import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from mynet import MyNet

exp_dict = {
    0:  ['A2C4',   0.00025, 0.0010, 0.004,  0],
    1:  ['A2C5_7', 0.00025, 0.0010, 0.0057, 0],
    2:  ['A2C7',   0.00025, 0.0010, 0.007,  0],
    3:  ['A2C8_7', 0.00025, 0.0010, 0.0087, 0],
    4:  ['A2C10',  0.00025, 0.0010, 0.01,   0],
    5:  ['A3C4',   0.00025, 0.0015, 0.004,  0],
    6:  ['A3C5_7', 0.00025, 0.0015, 0.0057, 0],
    7:  ['A3C7',   0.00025, 0.0015, 0.007,  0],
    8:  ['A3C8_7', 0.00025, 0.0015, 0.0087, 0],
    9:  ['A3C10',  0.00025, 0.0015, 0.01,   0],
    10: ['A4C4',   0.00025, 0.0020, 0.004,  0],
    11: ['A4C5_7', 0.00025, 0.0020, 0.0057, 0],
    12: ['A4C7',   0.00025, 0.0020, 0.007,  0],
    13: ['A4C8_7', 0.00025, 0.0020, 0.0087, 0],
    14: ['A4C10',  0.00025, 0.0020, 0.01,   0],
    15: ['A5C4',   0.00025, 0.0025, 0.004,  0],
    16: ['A5C5_7', 0.00025, 0.0025, 0.0057, 0],
    17: ['A5C7',   0.00025, 0.0025, 0.007,  0],
    18: ['A5C8_7', 0.00025, 0.0025, 0.0087, 0],
    19: ['A5C10',  0.00025, 0.0025, 0.01,   0],
    20: ['A6C4',   0.00025, 0.0030, 0.004,  0],
    21: ['A6C5_7', 0.00025, 0.0030, 0.0057, 0],
    22: ['A6C7',   0.00025, 0.0030, 0.007,  0],
    23: ['A6C8_7', 0.00025, 0.0030, 0.0087, 0],
    24: ['A6C10',  0.00025, 0.0030, 0.01,   0]
}

# Choose samples
sample = [i for i in range(25)]

# Manual flag
target_flags = [0, 1, 1, 0]

# Load experimental data
signals_exp = np.ndarray([0, 2001])
targets_exp = np.ndarray([4, 0])
for i in sample:
    temp = np.expand_dims(np.genfromtxt('exp/' + exp_dict[i][0] + '.txt',
                            delimiter=',', dtype=np.float32), axis=0)
    signals_exp = np.concatenate([signals_exp, temp])
    temp = np.expand_dims(np.array(exp_dict[i][1:5]).transpose(), axis=1)
    targets_exp = np.concatenate([targets_exp, temp], axis=1)

targets_exp = targets_exp[1:3, :]

# Convert to tensor
inputs_exp = torch.unsqueeze(torch.from_numpy(signals_exp[:, 241:]/1.409946942046675e-07).float(), 1)
targets_exp = torch.tensor(targets_exp).float()

if targets_exp.ndim == 1:
    targets_exp = torch.unsqueeze(targets_exp, 1)
else:
    targets_exp = torch.transpose(targets_exp, 0, 1)

# Load the trained CNN
model_name = '92678f6b'
model = torch.load('trained_model/' + model_name + '/my_model.pt')

model.eval()
with torch.no_grad():
    outputs_exp = model(inputs_exp)

outputs_exp = outputs_exp.numpy()
targets_exp = targets_exp.numpy()

v_range = np.array([[0.00025, 0.00025],  # short axis
                    [0.0005, 0.003],  # size
                    [0.003, 0.011],  # depth
                    [0, 0]])  # orientation
column_index = 0
for i in range(4):
    if target_flags[i]:
        outputs_exp[:, column_index] = outputs_exp[:, column_index] * (
                    v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
        column_index += 1

exp_error = abs((outputs_exp - targets_exp) / targets_exp) * 100
print('Experimental predictions are:')
for item in outputs_exp.tolist():
    item[0] = item[0] * 2
    print(item)
print('Experimental true values are:')
for item in targets_exp.tolist():
    item[0] = item[0] * 2
    print(item)
print('Experimental errors are:')
for item in exp_error.tolist():
    print(item)

# Calculate MAPE and MAE for experimental data
error = targets_exp - outputs_exp
abs_error = abs(error)
rel_error = error / targets_exp * 100
abs_rel_error = abs(rel_error)
MAPE = np.sum(abs_rel_error, 0) / abs_rel_error.shape[0]
MAE = np.sum(abs_error, 0) / abs_error.shape[0]
MAE = MAE * 1000
print('MAPE for experiment data is', MAPE)
print('MAE for experiment data is', MAE)

# Write the variables to file and process with MATLAB for paper quality figures
wb1 = xlsxwriter.Workbook('../../MATLAB/HDPE NDT/CNN_results_exp.xlsx')
sheet1 = wb1.add_worksheet()  # writes exp data prediction
sheet2 = wb1.add_worksheet()  # writes exp data actual value

for i in range(outputs_exp.shape[0]):
    for j in range(outputs_exp.shape[1]):
        sheet1.write_number(i, j, outputs_exp[i][j])
        sheet2.write_number(i, j, targets_exp[i][j])
wb1.close()
