%%% Plot for paper on 3D NDT crack quantification %%%
clc
clear all
close all

% Load data
results = 'CNN_results'; % Use _aug for the time being

test_pred = xlsread(results,'Sheet3');
test_actual = xlsread(results,'Sheet4');

%%
% Plot switches
Loss = 1;
Size = 1;
Location = 1;

test_error = test_actual - test_pred;
test_abs_error = abs(test_error);
test_rel_error = test_error ./ test_actual * 100;
test_absrel_error = abs(test_rel_error);
test_mape = sum(test_absrel_error, 1) / length(test_absrel_error);
test_mae = sum(test_abs_error, 1) / length(test_abs_error);
test_mae(1:2) = test_mae(1:2) * 1000;

% exp_error = exp_actual - exp_pred;
% exp_abs_error = abs(exp_error);
% exp_rel_error = exp_error ./ exp_actual * 100;
% exp_absrel_error = abs(exp_rel_error);
% exp_mape = sum(exp_absrel_error, 1) / length(exp_absrel_error);
% exp_mae = sum(exp_abs_error, 1) / length(exp_abs_error);
% exp_mae(1:2) = exp_mae(1:2) * 1000;

%%%%%%%%%%%%%%%%%%%%%%%%% Plot size %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Size
    figure
    hold on
    plot(test_actual(:,1)*2000,test_pred(:,1)*2000,'^','linewidth',7,'markersize',7,'color',[0.7, 0.3, 0.2]);
    plot(linspace(1,6,100),linspace(1,6,100),'--k','linewidth',5);
    xlabel('Actual Value (mm)')
    ylabel('CNN Prediction (mm)')
    legend('Testing data','Prediction = Actual', 'location', 'northwest')
    xlim([1, 6])
    ylim([1, 6])
    box on
    set(gca,'FontSize',40)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
end

%%%%%%%%%%%%%%%%%%%%%%%% Plot location %%%%%%%%%%%%%%%%%%%%%%%%%%%
if Location
    figure
    hold on
    plot(test_actual(:,2)*1000,test_pred(:,2)*1000,'^','linewidth',7,'markersize',7,'color',[0.1, 0.3, 0.7]);
    plot(linspace(3,11,100),linspace(3,11,100),'--k','linewidth',5);
    xlabel('Actual Value (mm)')
    ylabel('CNN Prediction (mm)')
    legend('Testing data','Prediction = Actual', 'location', 'northwest')
    xlim([3, 11])
    ylim([3, 11])
    xticks([3,5,7,9,11])
    yticks([3,5,7,9,11])
    box on
    set(gca,'FontSize',40)
    set(gca,'YColor','k')
    set(gca,'LineWidth',2);
    set(gcf,'Units','Inches');
    set(gcf,'Position',[2 0.2 12 10.2])
    set(gca,'DataAspectRatio', [1 1 1])
end