%%% Plot experimental results for HDPE CNN %%%
%%% Written on 04/14/2022 %%%

clc
clear all
close all

% Load data
results = 'CNN_results_exp';

exp_pred = xlsread(results,'Sheet1');
exp_actual = xlsread(results,'Sheet2');

% Calculate MAPE and MAE
exp_error = exp_actual - exp_pred;
exp_abs_error = abs(exp_error);
exp_rel_error = exp_error ./ exp_actual * 100;
exp_absrel_error = abs(exp_rel_error);
exp_mape = sum(exp_absrel_error, 1) / length(exp_absrel_error);
exp_mae = sum(exp_abs_error, 1) / length(exp_abs_error) * 1000;

% Plot the results
figure
hold on
box on
plot(exp_actual(:,1)*2000,exp_pred(:,1)*2000,'^','linewidth',15,'markersize',15,'color',[0.7, 0.3, 0.2])
plot(linspace(1.5,6.5,100),linspace(1.5,6.5,100),'--k','linewidth',8);
set(gca,'FontSize',44)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 12 10.2])
set(gca,'DataAspectRatio', [1 1 1])
xlabel('Actual Value (mm)')
ylabel('CNN Prediction (mm)')
legend('Experimental data','Prediction = Actual', 'location', 'northwest')
xlim([1.5, 6.5])
ylim([1.5, 6.5])
xticks([2, 3, 4, 5, 6])
yticks([2, 3, 4, 5, 6])

figure
hold on
box on
plot(exp_actual(:,2)*1000,exp_pred(:,2)*1000,'^','linewidth',15,'markersize',15,'color',[0.1, 0.3, 0.7])
plot(linspace(3,11,100),linspace(3,11,100),'--k','linewidth',8);
set(gca,'FontSize',44)
set(gca,'YColor','k')
set(gca,'LineWidth',2);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 12 10.2])
set(gca,'DataAspectRatio', [1 1 1])
xlabel('Actual Value (mm)')
ylabel('CNN Prediction (mm)')
legend('Experimental data','Prediction = Actual', 'location', 'northwest')
xlim([3, 11])
ylim([3, 11])
xticks([3, 5, 7, 9, 11])
yticks([3, 5, 7, 9, 11])