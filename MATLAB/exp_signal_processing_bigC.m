%%% Add dummy data point in experimental signal to match simulation %%%
%%% This is for C = 7, 8.7 and 10 %%%

clc
clear all
close all

sample = 'A5C10';

experiment = xlsread(['Exp data/', sample, '.xlsx']);
simulation_data = xlsread(['Exp data/A2C4_sim.xlsx']);

%%
% Choose a signal to use
column = 2;
scaling = 3.5;
experiment_data = experiment(:,column);
final_signal = zeros(1,2001);

time = [0:1e-8:2e-5];

start_index = 1570;
end_index = start_index + 3520;

final_signal(241:end) = experiment_data(start_index:2:end_index);

simulation_data(1:240) = 0;

%%
plot(final_signal/10^9/scaling,'linewidth',3)
hold on
plot(abs(simulation_data),'linewidth',3)
box on
ylabel('Displacement (m)')
xlabel('Time (s)')
legend('Experiment','Simulation')
set(gca,'FontSize',44)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 1.5*10. 1.37*7.5])

%%
% exp_out = 'C:\Users\sniu3\Documents\python_work\HDPE_CNN\exp\';
% writematrix(final_signal/10^9/scaling, [exp_out, sample, '.txt']);