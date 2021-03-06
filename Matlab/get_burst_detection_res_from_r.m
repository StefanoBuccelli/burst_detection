%% get results from r
clear
clc

%% load one channel spike detection
load('D:\Capocaccia\Basal\ptrain_20170519_01_nbasal_0005\ptrain_20170519_01_nbasal_0005_21.mat')
fs=1e4;
time_stamps_samples=find(peak_train);
time_stamps_s=time_stamps_samples/fs;

% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end
cd ..

fileID= fopen('spikes_s.txt','w');
for curr_sample=1:length(time_stamps_s)
    fprintf(fileID,'%f\n',time_stamps_s(curr_sample));
end
fclose(fileID);

%% if you load from an existing file with timestamps
% load('spikes_s.mat')
% fs=1e4;
% time_stamps_samples=Spikes_s;
% time_stamps_s=time_stamps_samples/fs;

%% load burst detection from SpiNNaker

%% load burst detection from Matlab functions
load('D:\Capocaccia\Basal\10khz_data_bursts\data_basal\ch 21.mat')
%limit to first 300s
burst_new(burst_new(:,1)>=300*fs,:)=[];
burst_new_s=burst_new/fs;
%% load results from r session as tables

resultCMA = importfile('result_CMA.csv');
resultHenning = importfile('result_hennig_method.csv');
resultPS = importfile('result_PS_method.csv');
resultRS = importfile('result_RS_method.csv');
resultLogISI = importfile('result_pasquale_method.csv');
resultMI = importfile('result_MI_method.csv');
resultHSMM = importfile('result_HSMM_method.csv');

%% plot spikes and burst detection
figure
% plot spikes first
% plot(time_stamps_samples,ones(size(time_stamps_samples)),'b.')
for curr_spike=1:length(time_stamps_s)
    curr_sample=time_stamps_s(curr_spike);
    plot([curr_sample curr_sample],[0 1],'b')
    hold on
end
hold on
% plot bursts from r
for curr_burst=1:height(resultCMA)
    start_sample=time_stamps_s(resultCMA.beg(curr_burst));
    stop_sample=time_stamps_s(resultCMA.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2),'ro-')
end
for curr_burst=1:height(resultHenning)
    start_sample=time_stamps_s(resultHenning.beg(curr_burst));
    stop_sample=time_stamps_s(resultHenning.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.2,'go-')
end
for curr_burst=1:height(resultPS)
    start_sample=time_stamps_s(resultPS.beg(curr_burst));
    stop_sample=time_stamps_s(resultPS.end1(curr_burst));   
    plot([start_sample stop_sample],zeros(1,2)+.4,'mo-')
end
for curr_burst=1:height(resultRS)
    start_sample=time_stamps_s(resultRS.beg(curr_burst));
    stop_sample=time_stamps_s(resultRS.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.6,'ko-')
end
% plot bursts from Paquale LogISI
for curr_burst=1:size(resultLogISI,1)
    start_sample=time_stamps_s(resultLogISI.beg(curr_burst));
    stop_sample=time_stamps_s(resultLogISI.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.8,'yo-')
end
% plot bursts from MI
for curr_burst=1:size(resultMI,1)
    start_sample=time_stamps_s(resultMI.beg(curr_burst));
    stop_sample=time_stamps_s(resultMI.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.9,'g*-')
end
% plot bursts from HSMM
for curr_burst=1:size(resultHSMM,1)
    start_sample=time_stamps_s(resultHSMM.beg(curr_burst));
    stop_sample=time_stamps_s(resultHSMM.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+1,'r*-')
end
% plot bursts from matlab (Chiappalone)
for curr_burst=1:size(burst_new_s,1)
    start_sample=time_stamps_s(time_stamps_s==burst_new_s(curr_burst,1));
    stop_sample=time_stamps_s(time_stamps_s==burst_new_s(curr_burst,2));
    plot([start_sample stop_sample],zeros(1,2)+1.1,'co-')
end
% improve graphics
ylim([-.1 1.2])
title('spike detection comparison')
xlabel('Time [s]')
yticks([0 .2 .4 .6 .8 .9 1 1.1])
yticklabels({'CMA','Henning','PS','RS','LogISI','MI','HSMM','CH'})

% xlim([1.5 2.5])

%% comparing number of bursts detected
figure
bar([height(resultCMA) height(resultHenning) height(resultLogISI) height(resultPS) height(resultRS) height(resultMI) height(resultHSMM)])
xticklabels({'CMA','ISI Rank','Log Isi','PS','RS','MI','HSMM'})
ylabel('# of bursts')
title('comparing number of bursts detected')

%% comparing mean burst duration
figure
bar([mean(resultCMA.durn) mean(resultHenning.durn) mean(resultLogISI.durn) mean(resultPS.durn) mean(resultRS.durn) mean(resultMI.durn) mean(resultHSMM.durn)])
xticklabels({'CMA','ISI Rank','Log Isi','PS','RS','MI','HSMM'})
ylabel('Mean burst duration (s)')
title('comparing mean burst duration')


