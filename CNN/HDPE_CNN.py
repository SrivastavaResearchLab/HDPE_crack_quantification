import torch
import torch.nn as nn
import numpy as np
import uuid
import os
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
from mynet import MyNet

torch.manual_seed(123)
np.random.seed(123)

'''Hyper-parameters'''
# Training
num_epochs = 2000
learning_rate = 0.0005

# Dataset dictionary
train_dataset_dict = {
    1: '1502_size_depth'
}

test_dataset_dict = {
    1: '92_size_depth'
}

# Choose a training dataset
dataset = 1

# Load the data
signals = pd.read_excel('train/' + train_dataset_dict[dataset] + '.xlsx',
                        header=None).values
signals_info = pd.read_excel('train/' + train_dataset_dict[dataset] + '_info.xlsx',
                             header=None).values

# Load test data
signals_test = pd.read_excel('test/' + test_dataset_dict[dataset] + '.xlsx',
                             header=None).values
signals_test_info = pd.read_excel('test/' + test_dataset_dict[dataset] + '_info.xlsx',
                                  header=None).values

# Remove nan rows from signals
signals = signals[~np.isnan(signals).any(axis=1)]
signals_test = signals_test[~np.isnan(signals_test).any(axis=1)]

# Remove nan columns from info
signals_info = signals_info[:, ~np.isnan(signals_info).any(axis=0)]
signals_test_info = signals_test_info[:, ~np.isnan(signals_test_info).any(axis=0)]

# Find the non-zeros rows in the info and delete the rest
target_flags = signals_info.any(axis=1)
targets_train = signals_info[target_flags == True]
targets_test = signals_test_info[target_flags == True]

# Normalize all the features into the region [0, 1]
row_index = 0
# Normalize the targets
v_range = np.array([[0.00025, 0.00025],  # short axis
                    [0.0005,    0.003],  # size
                    [0.003,     0.011],  # depth
                    [0,            0]])  # orientation
for i in range(4):
    if target_flags[i]:
        targets_train[row_index, :] = (targets_train[row_index, :] - v_range[i, 0]) / \
                                      (v_range[i, 1] - v_range[i, 0])
        targets_test[row_index, :] = (targets_test[row_index, :] - v_range[i, 0]) / \
                                      (v_range[i, 1] - v_range[i, 0])
        row_index += 1

# Transform the data into tensor
# Additional notes: the inputs are normalized by the maximum value in the signal and the data type
#                   is changed from float64 to float32
inputs_train = torch.unsqueeze(torch.from_numpy(abs(signals[:, 241:]/max(signals[0, :]))).float(), 1)
targets_train = torch.from_numpy(targets_train).float()

inputs_test = torch.unsqueeze(torch.from_numpy(abs(signals_test[:, 241:]/max(signals[0, :]))).float(), 1)
targets_test = torch.from_numpy(targets_test).float()

print(max(signals[0, :]))

# Set the output neuron number accordingly
fc_out = sum(target_flags)

if targets_train.ndim == 1:
    targets_train = torch.unsqueeze(targets_train, 1)
    targets_test = torch.unsqueeze(targets_test, 1)
else:
    targets_train = torch.transpose(targets_train, 0, 1)
    targets_test = torch.transpose(targets_test, 0, 1)

# CNN model
model = MyNet(fc_out)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Variable to store epoch and loss for plotting
loss_train = []

# Train the model
for epoch in range(num_epochs):

    # Forward pass
    outputs_train = model(inputs_train)
    outputs_test = model(inputs_test)
    loss = criterion(outputs_train, targets_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_train.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch + 1, num_epochs, loss.item()))

# Generate a random path and directory to store temp-trained model
temp_name = str(uuid.uuid4())[-8:] + '/'
model_path = 'trained_model/' + temp_name
os.mkdir(model_path)

# Test the trained model
model.eval()
with torch.no_grad():
    outputs_test = model(inputs_test)
    outputs_train = model(inputs_train)

outputs_test = outputs_test.detach().numpy()
targets_test = targets_test.numpy()
outputs_train = outputs_train.detach().numpy()
targets_train = targets_train.numpy()

# De-normalize the data for visualization
column_index = 0
for i in range(4):
    if target_flags[i]:
        outputs_test[:, column_index] = outputs_test[:, column_index] * (
                v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
        outputs_train[:, column_index] = outputs_train[:, column_index] * (
                v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
        targets_train[:, column_index] = targets_train[:, column_index] * (
                v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
        targets_test[:, column_index] = targets_test[:, column_index] * (
                v_range[i, 1] - v_range[i, 0]) + v_range[i, 0]
        column_index += 1

# Calculate MAPE for test data
abs_error = targets_test - outputs_test
rel_error = abs_error / targets_test * 100
abs_rel_error = abs(rel_error)
MAPE = np.sum(abs_rel_error, 0) / abs_rel_error.shape[0]
print('MAPE for test data is', MAPE)

# Calculate MAPE for training data
abs_error = targets_train - outputs_train
rel_error = abs_error / targets_train * 100
abs_rel_error = abs(rel_error)
MAPE = np.sum(abs_rel_error, 0) / abs_rel_error.shape[0]
print('MAPE for training data is', MAPE)

# Plot training epochs
fig, ax = plt.subplots(1, 1)
ax.plot(loss_train)
ax.set_xlabel('Training Epochs')
ax.set_yscale('log')
ax.grid(linestyle='--')
plt.show()
np.save(model_path + 'loss.npy', np.array(loss_train))

# A rolling index to indicate which output to be plotted
plot_index = 0

if target_flags[1]:
    # Plot P-A for size
    PAline = np.linspace(1, 6, num=30)
    fig, ax = plt.subplots()
    ax.scatter(targets_test[:, plot_index] * 2000, outputs_test[:, plot_index] * 2000,
               s=150, marker='s', color='red')
    ax.plot(PAline, PAline, 'k--', linewidth=5)
    ax.set_xlabel('Actual Value (mm)', fontsize=30)
    ax.set_ylabel('Predicted Value (mm)', fontsize=30)
    ax.set_xticks(np.arange(1, 7))
    ax.set_yticks(np.arange(1, 7))
    ax.set_xlim([1, 6])
    ax.set_ylim([1, 6])
    ax.tick_params(labelsize=25)
    ax.set_aspect('equal', adjustable='box')
    fig.set_size_inches(8, 8)
    plt.title('Crack length prediction', fontsize=30)
    plt.legend(['Pred=Actual', 'Testing data'], fontsize=22, frameon=False)
    # plt.savefig('Size_pred.png', format='png', dpi=600)
    plt.show()
    plot_index += 1

if target_flags[2]:
    # Plot P-A for depth
    PAline = np.linspace(3, 11, num=30)
    fig, ax = plt.subplots()
    ax.scatter(targets_test[:, plot_index] * 1000, outputs_test[:, plot_index] * 1000,
               s=150, marker='s', color='red')
    ax.plot(PAline, PAline, 'k--', linewidth=5)
    ax.set_xlabel('Actual Value (mm)', fontsize=30)
    ax.set_ylabel('Predicted Value (mm)', fontsize=30)
    ax.set_xticks(np.arange(3, 12))
    ax.set_yticks(np.arange(3, 12))
    ax.set_xlim([3, 11])
    ax.set_ylim([3, 11])
    ax.tick_params(labelsize=25)
    ax.set_aspect('equal', adjustable='box')
    fig.set_size_inches(8, 8)
    plt.title('Crack location prediction', fontsize=30)
    plt.legend(['Pred=Actual', 'Testing data'], fontsize=22, frameon=False)
    # plt.savefig('Depth_pred.png', format='png', dpi=600)
    plt.show()
    plot_index += 1

# Save the model and write the model parameters to file
torch.save(model, model_path + 'my_model.pt')

with open(model_path + "model_param.txt", "w") as f:
    # Writing data to a file
    f.write("---------------------------------------------------------------")
    f.write("\n Training dataset: " + train_dataset_dict[dataset])
    f.write("\n Testing dataset: " + test_dataset_dict[dataset])
    f.write("\n Training epochs: " + str(num_epochs))
    f.write("\n Learning rate: " + str(learning_rate))
    f.write("\n---------------------------------------------------------------")
    f.write("\n CNN configuration:")
    f.write("\n" + str(model))
    f.write("\n---------------------------------------------------------------")

'''
# Write the variables to file and process with MATLAB for paper quality figures
wb1 = xlsxwriter.Workbook("../../MATLAB/HDPE NDT/CNN_results.xlsx")
sheet1 = wb1.add_worksheet() # writes training data prediction
sheet2 = wb1.add_worksheet() # writes training data actual value
sheet3 = wb1.add_worksheet() # writes test data prediction
sheet4 = wb1.add_worksheet() # writes test data actual value
sheet5 = wb1.add_worksheet() # writes training epoch and loss data

for i in range(outputs_train.shape[0]):
    for j in range(outputs_train.shape[1]):
        sheet1.write_number(i, j, outputs_train[i][j])
        sheet2.write_number(i, j, targets_train[i][j])
for i in range(outputs_test.shape[0]):
    for j in range(outputs_test.shape[1]):
        sheet3.write_number(i, j, outputs_test[i][j])
        sheet4.write_number(i, j, targets_test[i][j])
for i in range(num_epochs):
    sheet5.write_number(0, i, loss_train[i])

wb1.close()
'''