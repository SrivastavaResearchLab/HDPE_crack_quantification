%%% This script is for generating input pulse signal %%%
clc
clear all
close all

freq = 1*10^6; % Hz
N = 2.5; % number of cycles
time = linspace(0,N/freq,101)';

signal = cos(2*pi*freq*time).*(1-cos(2*pi*freq/N*time));
% signal = sin(2*pi*time*freq);
% data = xlsread('full_5MHz.xlsx');
% plot(data(:,1),data(:,2))
% Fs = 1/(time(2)-time(1));

% Y = fft(signal);
% P2 = abs(Y/51);
% P1 = P2(1:51/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% f = Fs*(0:(51/2))/51;
% plot(f,P1) 

plot(time*1000000,signal,'k','linewidth',5)
grid on
xlabel('Time (\mus)')
ylabel('Amplitude')
xlim([0, 2.5])
set(gca,'FontSize',60)
set(gca,'LineWidth',5);
set(gcf,'Units','Inches');
set(gcf,'Position',[2 0.2 1.5*10. 1.37*7.5])
set(gcf,'DefaultTextColor','black')

% write to file
% output = [time,signal]';
% outfile = fopen('signal.inp','wt');
% fprintf(outfile,'%d, %d,\n',output);
% fclose(outfile);
% output = [time]';
% outfile = fopen('signal.inp','wt');
% fprintf(outfile,'%d \n',output);
% fclose(outfile);