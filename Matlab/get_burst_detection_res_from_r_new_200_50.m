%% get results from r
clear
close all
clc

signal_color=jet(10);
%% load one channel spike detection
ch_num=12;
phase_dir='D:\Capocaccia\New_training_test_sets\Test_set\20171222\20171222_FilteredData\20171222_20512_PeakDetectionMAT_PLP2ms_RP1ms\ptrain_20171222_20512_01_nbasal_0001';
burst_dir='D:\Capocaccia\New_training_test_sets\Test_set\20171222\20171222_FilteredData\20171222_20512_BurstDetectionMAT_5-100msec\20171222_20512_BurstDetectionFiles';
load([phase_dir '\ptrain_20171222_20512_01_nbasal_0001_' num2str(ch_num) '.mat'])
load(['D:\Capocaccia\New_training_test_sets\Test_set\20171222_20512_01_nbasal_0001_' num2str(ch_num) '_raw_hpf.mat'])
% load ground truth visual inspection
load(['D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\burst_start_stop_ch' num2str(ch_num) '.mat'])

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

%% load burst detection CH from Matlab functions
load([burst_dir '\burst_detection_20171222_20512_01_nbasal_0001.mat'])
%limit to first 300s
burst_new_s=burst_detection_cell{ch_num,1}(1:end-1,1:2)./fs;
burst_new_dur=burst_new_s(:,2)-burst_new_s(:,1);
% burst_new(burst_new(:,1)>=300*fs,:)=[];
% burst_new_s=burst_new/fs;

%% load detection from cochlea 200ms
cochlea_res= import_results_cochlea(['D:\Capocaccia\New_training_test_sets\results_from_cochlea\classification_done_ch' num2str(ch_num) 'raw.csv']);
shift=0.2;
[start_stop_cochlea_dur,start_stop_cochlea] = get_start_stop_cochlea(cochlea_res,shift);

cochlea_res_50= import_results_cochlea(['D:\Capocaccia\New_training_test_sets\results_from_cochlea\classification_done_ch' num2str(ch_num) 'raw_50ms.csv']);
shift=0.05;
[start_stop_cochlea_dur_50,start_stop_cochlea_50] = get_start_stop_cochlea(cochlea_res_50,shift);

%% start stop visual inspection
start_stop_vi_duration=visual_inspection(:,2)-visual_inspection(:,1);

%% load results from r session as tables
cd(['D:\Capocaccia\New_training_test_sets\results_from_r\ch_' num2str(ch_num)])
mkdir('Figures')
resultCMA = importfile('result_CMA.csv');
resultHenning = importfile('result_hennig_method.csv');
resultPS = importfile('result_PS_method.csv');
resultRS = importfile('result_RS_method.csv');
resultLogISI = importfile('result_pasquale_method.csv');
resultMI = importfile('result_MI_method.csv');
resultHSMM = importfile('result_HSMM_method.csv');

%% initialize signals for crosscorr
time_s=(1:1:length(peak_train))./fs;
signal_cma=zeros(1,length(peak_train));
signal_henning=zeros(1,length(peak_train));
signal_ps=zeros(1,length(peak_train));
signal_rs=zeros(1,length(peak_train));
signal_logisi=zeros(1,length(peak_train));
signal_mi=zeros(1,length(peak_train));
signal_hsmm=zeros(1,length(peak_train));
signal_ch=zeros(1,length(peak_train));
signal_nas=zeros(1,length(peak_train));
signal_nas_50=zeros(1,length(peak_train));
signal_vi=zeros(1,length(peak_train));

signal_names=fliplr({'CMA','Hen','PS','RS','LogISI','MI','HSMM','CH','NAS','VI'});

%% plot spikes and burst detection
burst_spikes_overlap=figure;
% plot spikes first
% plot(time_stamps_samples,ones(size(time_stamps_samples)),'b.')
for curr_spike=1:length(time_stamps_s)
    curr_sample=time_stamps_s(curr_spike);
    plot([curr_sample curr_sample],[0 1.8],'Color',[.9 .9 .9],'LineWidth',0.2)
    hold on
end
hold on
% plot bursts from r
for curr_burst=1:height(resultCMA)
    start_sample_s=time_stamps_s(resultCMA.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultCMA.end1(curr_burst));
    plot([start_sample_s stop_sample_s],zeros(1,2),'Marker','o','MarkerEdgeColor',signal_color(1,:),'Color',signal_color(1,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_cma(start_sample:stop_sample)=1;
end
for curr_burst=1:height(resultHenning)
    start_sample_s=time_stamps_s(resultHenning.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultHenning.end1(curr_burst));
    plot([start_sample_s stop_sample_s],zeros(1,2)+.2,'Marker','o','MarkerEdgeColor',signal_color(2,:),'Color',signal_color(2,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_henning(start_sample:stop_sample)=1;
end
for curr_burst=1:height(resultPS)
    start_sample_s=time_stamps_s(resultPS.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultPS.end1(curr_burst));   
    plot([start_sample_s stop_sample_s],zeros(1,2)+.4,'Marker','o','MarkerEdgeColor',signal_color(3,:),'Color',signal_color(3,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_ps(start_sample:stop_sample)=1;
end
for curr_burst=1:height(resultRS)
    start_sample_s=time_stamps_s(resultRS.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultRS.end1(curr_burst));
    plot([start_sample_s stop_sample_s],zeros(1,2)+.6,'Marker','o','MarkerEdgeColor',signal_color(4,:),'Color',signal_color(4,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_rs(start_sample:stop_sample)=1;
end
% plot bursts from Paquale LogISI
for curr_burst=1:size(resultLogISI,1)
    start_sample_s=time_stamps_s(resultLogISI.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultLogISI.end1(curr_burst));
    plot([start_sample_s stop_sample_s],zeros(1,2)+.8,'Marker','o','MarkerEdgeColor',signal_color(5,:),'Color',signal_color(5,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_logisi(start_sample:stop_sample)=1;
end
% plot bursts from MI
for curr_burst=1:size(resultMI,1)
    start_sample_s=time_stamps_s(resultMI.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultMI.end1(curr_burst));
    plot([start_sample_s stop_sample_s],zeros(1,2)+1,'Marker','o','MarkerEdgeColor',signal_color(6,:),'Color',signal_color(6,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_mi(start_sample:stop_sample)=1;
end
% plot bursts from HSMM
for curr_burst=1:size(resultHSMM,1)
    start_sample_s=time_stamps_s(resultHSMM.beg(curr_burst));
    stop_sample_s=time_stamps_s(resultHSMM.end1(curr_burst));
    plot([start_sample_s stop_sample_s],zeros(1,2)+1.2,'Marker','o','MarkerEdgeColor',signal_color(7,:),'Color',signal_color(7,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_hsmm(start_sample:stop_sample)=1;
end
% plot bursts from matlab (Chiappalone)
for curr_burst=1:size(burst_new_s,1)
    start_sample_s=time_stamps_s(time_stamps_s==burst_new_s(curr_burst,1));
    stop_sample_s=time_stamps_s(time_stamps_s==burst_new_s(curr_burst,2));
    plot([start_sample_s stop_sample_s],zeros(1,2)+1.4,'Marker','o','MarkerEdgeColor',signal_color(8,:),'Color',signal_color(8,:))
    %
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    signal_ch(start_sample:stop_sample)=1;
end
% plot bursts from cochlea
for curr_burst=1:size(start_stop_cochlea,1)
    start_sample_s=start_stop_cochlea(curr_burst,1);
    stop_sample_s=start_stop_cochlea(curr_burst,2);
%     plot([start_sample_s stop_sample_s],zeros(1,2)+1.6,'Marker','o','MarkerEdgeColor',signal_color(9,:),'Color',signal_color(9,:))
    %
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    signal_nas(start_sample:stop_sample)=1;
end
% get bursts from cochlea 50ms
for curr_burst=1:size(start_stop_cochlea_50,1)
    start_sample_s=start_stop_cochlea_50(curr_burst,1);
    stop_sample_s=start_stop_cochlea_50(curr_burst,2);
    plot([start_sample_s stop_sample_s],zeros(1,2)+1.6,'Marker','o','MarkerEdgeColor',signal_color(9,:),'Color',signal_color(9,:))
    %
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    signal_nas_50(start_sample:stop_sample)=1;
end
% plot bursts from visual inspection
for curr_burst=1:size(visual_inspection,1)
    start_sample_s=visual_inspection(curr_burst,1);
    stop_sample_s=visual_inspection(curr_burst,2);
    plot([start_sample_s stop_sample_s],zeros(1,2)+1.8,'Marker','o','MarkerEdgeColor',signal_color(10,:),'Color',signal_color(10,:))
    %
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    signal_vi(start_sample:stop_sample)=1;
end

% improve graphics
ylim([-.1 1.9])
title('spike detection comparison')
xlabel('Time [s]')
yticks([0 .2 .4 .6 .8 1 1.2 1.4 1.6 1.8])
yticklabels(fliplr(signal_names))
savefig(burst_spikes_overlap,'Figures\burst_spikes_overlap')
% xlim([1.5 2.5])

%% comparing number of bursts detected
burst_numb_dur=figure;
subplot(3,1,1)
n_bursts=[height(resultCMA) height(resultHenning) height(resultPS) height(resultRS) height(resultLogISI) height(resultMI) height(resultHSMM) size(burst_new_s,1) size(start_stop_cochlea_50,1) size(visual_inspection,1)];
b=bar(n_bursts(1:end-1));
b.FaceColor = 'flat';
b.CData = signal_color(1:end-1,:);
hold on
plot([0 10],[size(visual_inspection,1) size(visual_inspection,1)],'k--')
xticklabels(fliplr(signal_names(2:end)))
ylabel('# of bursts')
title('comparing number of bursts detected 50ms')

subplot(3,1,2)
b=bar(n_bursts(1:end-1)./n_bursts(end));
b.FaceColor = 'flat';
b.CData = signal_color(1:end-1,:);
hold on
plot([0 10],[1 1],'k--')
xticklabels(fliplr(signal_names(2:end)))
title('ratio #burst/VI')
ylabel('ratio #burst/VI')

%% comparing mean burst duration
mean_vals=[mean(resultCMA.durn) mean(resultHenning.durn) mean(resultPS.durn) mean(resultRS.durn) mean(resultLogISI.durn) mean(resultMI.durn) mean(resultHSMM.durn) mean(burst_new_dur) mean(start_stop_cochlea_dur_50) mean(start_stop_vi_duration)];
std_vals=[std(resultCMA.durn) std(resultHenning.durn) std(resultPS.durn) std(resultRS.durn) std(resultLogISI.durn) std(resultMI.durn) std(resultHSMM.durn) std(burst_new_dur) std(start_stop_cochlea_dur_50) std(start_stop_vi_duration)];
se_vals=std_vals./sqrt(n_bursts);

% figure
% bar(mean_vals)
% hold on
% errorbar(mean_vals,std_vals,'.k')
% xticklabels(fliplr(signal_names))
% ylabel('Mean burst duration (s)')
% title('comparing mean burst duration +- STD')

subplot(3,1,3)
b=bar(mean_vals);
b.FaceColor = 'flat';
b.CData = signal_color;
hold on
errorbar(mean_vals,se_vals,'.k')
xticklabels(fliplr(signal_names))
ylabel('Mean burst duration (s)')
title('comparing mean burst duration +-SE')
savefig(burst_numb_dur,'Figures\burst_numb_dur_50ms')

%% correlation between burst events, first: create 0-1 signals with burst events
all_signals=[signal_vi; signal_nas_50 ;signal_ch; signal_hsmm; signal_mi; signal_logisi; signal_rs; signal_ps; signal_henning; signal_cma]';
corr_matrix=corrcoef(all_signals);
pc_imagesc=figure;
imagesc(corr_matrix)
yticklabels(signal_names)
xticklabels(signal_names)
colorbar
title('Pearson coefficient')
savefig(pc_imagesc,'Figures\PC_imagesc_50ms')

%% correlation VI vs all
for curr_signal=1:10
    [xc(curr_signal,:),lags(curr_signal,:)]=xcorr(signal_vi,all_signals(:,curr_signal),1e5,'coef');
end

%% correlation VI vs all
cc_overlap=figure;
for curr_signal=1:10
    plot(lags(curr_signal,:)/fs,xc(curr_signal,:),'Color',signal_color(11-curr_signal,:))
    hold on
end
legend(signal_names)
title('cross correlation between VI and the other methods')
xlabel('Time [s]')
savefig(cc_overlap,'Figures\cc_overlap_50ms')

%% subplots
sub_cc=figure;
for curr_signal=1:10
   h(curr_signal)=subplot(2,5,curr_signal);
   plot(lags(curr_signal,:)/fs,xc(curr_signal,:),'LineWidth',3,'Color',signal_color(11-curr_signal,:))
   max_peak=max(xc(curr_signal,:));
   loc_max_peak=find(xc(curr_signal,:)==max_peak,1);
   max_lag=lags(curr_signal,loc_max_peak)/fs;
   title([signal_names{curr_signal} ', p:' num2str(max_peak,'%.2f') ', L:' num2str(max_lag,'%.2f')])
   xlabel('Time [s]')
end
linkaxes(h,'xy')
savefig(sub_cc,'Figures\sub_cc_50')

%% burst + raw + hpf
figure_burst_raw_hpf=figure;
h_b(1)=subplot(9,1,[1 7]);
for curr_signal=1:10
    plot(time_s,all_signals(:,11-curr_signal)+1.2*curr_signal,'Color',signal_color(curr_signal,:))
    hold on
end
peak_train_full=full(peak_train);
peak_train_full(peak_train_full>0)=1;
plot(time_s,peak_train_full,'Color',[.9 .9 .9])
yticks([.2+0.5 1.2*(1:10)+0.5])
yticklabels({'Spikes','CMA','Henning','PS','RS','LogISI','MI','HSMM','CH','NAS','VI'})
xticks([])
h_b(2)=subplot(9,1,8);
plot(time_s,data_raw,'Color','k')
xticks([])
h_b(3)=subplot(9,1,9);
plot(time_s,data_hpf,'Color','k')
xlabel('Time [s]')
linkaxes(h_b,'x')
savefig(figure_burst_raw_hpf,'Figures\figure_burst_raw_hpf_50ms')

%% correlation all vs all
clear xc lags
for curr_signal_2=1:10
    disp(curr_signal_2)
    for curr_signal=1:10
        [xc(curr_signal,curr_signal_2,:),lags(curr_signal,curr_signal_2,:)]=xcorr(all_signals(:,curr_signal),all_signals(:,curr_signal_2),1e5,'coef');
        max_peak(curr_signal,curr_signal_2)=max(xc(curr_signal,curr_signal_2,:));
        loc_max_peak=find(xc(curr_signal,curr_signal_2,:)==max_peak(curr_signal,curr_signal_2),1);
        max_lag(curr_signal,curr_signal_2)=lags(curr_signal,curr_signal_2,loc_max_peak)/fs;
    end
end
%%
cc_max_lag_imagesc=figure;
subplot(1,2,1)
imagesc(max_peak)
title('max peak')
xticks(1:10)
yticks(1:10)
yticklabels(signal_names)
xticklabels(signal_names)
colorbar
subplot(1,2,2)
imagesc(max_lag)
title('lag at max peak')
xticks(1:10)
yticks(1:10)
yticklabels(signal_names)
xticklabels(signal_names)
colorbar
savefig(cc_max_lag_imagesc,'Figures\cc_max_lag_imagesc_50ms')


function [start_stop_cochlea_dur,start_stop_cochlea] = get_start_stop_cochlea(cochlea_res,shift)
cochlea_res(cochlea_res.class=='NO SPIKES',:)=[];
cochlea_res.class(cochlea_res.class=='NORMAL')='0';
cochlea_res.class(cochlea_res.class=='ABNORMAL')='1';
cochlea_res.class=str2double(cochlea_res.class);
cochlea_res.Time=cochlea_res.Time-cochlea_res.Time(1); %restart from the first normal value

cochlea_res.Time_stretch=linspace(0,299.9,length(cochlea_res.Time))'; %% I removed 200ms which is roughtly the duration of a window
% cochlea_res.Time_stretch=(0.2:0.2:298.8)'; %% totally wrong at the end
cochelea_burst_loc=cochlea_res.Time_stretch(cochlea_res.class==1);

%%
falling_edge=1; %% initialize to find rising edge first
rising_edge=0;
new_rising=0;
start_stop_cochlea=[];
for curr_time=2:length(cochlea_res.Time)
    % if falling_edge==1 look for rising edge
    if falling_edge==1
        if (cochlea_res.class(curr_time)-cochlea_res.class(curr_time-1))==1
            rising_edge=1;
            falling_edge=0;
            new_rising=new_rising+1;
            start_stop_cochlea(new_rising,1)=cochlea_res.Time_stretch(curr_time)-shift;%subtract 100ms from start assuming original timing is the central
            %            start_stop_cochlea(new_rising,1)=cochlea_res.Time_stretch(curr_time);
        else
            rising_edge=0;
        end
    else  %% look for falling edge
        if (cochlea_res.class(curr_time)-cochlea_res.class(curr_time-1))==(-1)
            falling_edge=1;
            start_stop_cochlea(new_rising,2)=cochlea_res.Time_stretch(curr_time)-shift;%subtract 100ms from end assuming original timing is the central
            %            start_stop_cochlea(new_rising,2)=cochlea_res.Time_stretch(curr_time);
        else
            falling_edge=0;
        end
    end
end
start_stop_cochlea_dur=start_stop_cochlea(:,2)-start_stop_cochlea(:,1);
end
