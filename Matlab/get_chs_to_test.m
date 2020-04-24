%% get results from r
clear
close all
clc
fs=1e4;

parent_dir='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all\';
parent_dir_raw='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_raw\';
cd(parent_dir)

% %%
% all_names_={'20171222_20512';
% '20180226_20508';
% '20180226_20512';
% '20180226_27557';
% '20180227_28314';
% '20180227_33719';
% '20180301_33716'};
%
% %%
%
% for curr_name=1 : length(all_names_)
%     name_exp=all_names_{curr_name,1};
%     curr_dir_peak_train=[name_exp '_HFP300_3000\' name_exp '_PeakDetectionMAT_PLP2ms_RP1ms\ptrain_' name_exp '_01_nbasal_0001'];
%
%     load([parent_dir name_exp '_HFP300_3000\' name_exp '_SpikeAnalysis\' name_exp '_MeanFiringRate - thresh0.01\mfr_ptrain_' name_exp '_01_nbasal_0001.mat']);
%     load([parent_dir name_exp '_HFP300_3000\' name_exp '_BurstAnalysis\' name_exp '_MeanStatReportSPIKEinBURST\MStReportSpinB_' name_exp '_01_nbasal_0001.mat']);
%     load([parent_dir name_exp '_HFP300_3000\' name_exp '_BurstAnalysis\' name_exp '_MeanStatReportBURST\MStReportB_' name_exp '_01_nbasal_0001.mat']);
%     %% populate training struct
%     training.names{curr_name,1}=name_exp;
%     training.ibr{curr_name,1}=[SRMspike(:,1) SRMspike(:,5)];
%     training.br{curr_name,1}=[SRMburst(:,1) SRMburst(:,3)];
%     training.mfr{curr_name,1}=mfr_table;
%
% end
% train_tab=struct2table(training);
% train_tab.Properties.RowNames = train_tab.names;
% train_tab.names =[];
%
% %% plot MFR
% for curr_name=1:8
%     figure
%     subplot(3,1,1)
%     plot(train_tab.mfr{curr_name,1}(:,1),train_tab.mfr{curr_name,1}(:,2))
%     title(train_tab.Properties.RowNames{curr_name},'Interpreter','none')
%     ylabel('MFR')
%     subplot(3,1,2)
%     plot(train_tab.ibr{curr_name,1}(:,1),train_tab.ibr{curr_name,1}(:,2))
%     ylabel('IBR')
%     subplot(3,1,3)
%     plot(train_tab.br{curr_name,1}(:,1),train_tab.br{curr_name,1}(:,2))
%     ylabel('Burst rate')
% end

load('train_tab.mat')
cd ..
signal_names=fliplr({'CMA','Hen','PS','RS','LogISI','MI','HSMM','CH','NAS','VI'});
signal_color=jet(10);
%% load one channel spike detection
for curr_ch = 1:2
    for curr_exp=1 : height(train_tab)
        name_exp=train_tab.Properties.RowNames{curr_exp};
        if curr_ch==1
            ch_indx=train_tab.Ch1(curr_exp);
        else
            ch_indx=train_tab.Ch2(curr_exp);
        end
        curr_dir_peak_train=[name_exp '_HFP300_3000\' name_exp '_PeakDetectionMAT_PLP2ms_RP1ms\ptrain_' name_exp '_01_nbasal_0001'];
        load([parent_dir curr_dir_peak_train '\ptrain_' name_exp '_01_nbasal_0001_' num2str(ch_indx) '.mat'])
        %
        %% load burst detection CH from Matlab functions
        curr_dir_burst_det=[name_exp '_HFP300_3000\' name_exp '_BurstDetectionMAT_5-100msec\' name_exp '_BurstDetectionFiles'];
        load([parent_dir curr_dir_burst_det '\burst_detection_' name_exp '_01_nbasal_0001.mat'])
        %limit to first 300s
        burst_CH_s=burst_detection_cell{ch_indx,1}(1:end-1,1:2)./fs;
        burst_CH_dur=burst_CH_s(:,2)-burst_CH_s(:,1);
        
        %% load raw and hpf
        nome_exp_no_=name_exp;
        nome_exp_no_(9)=[];
        load([parent_dir_raw nome_exp_no_ '\' nome_exp_no_ '_Mat_files\' nome_exp_no_ '_01_nbasal_0001\' nome_exp_no_ '_01_nbasal_0001_' num2str(ch_indx) '.mat'])
        data_raw=data;
        clear data
        load([parent_dir_raw nome_exp_no_ '\' nome_exp_no_ '_FilteredData\' nome_exp_no_ '_Mat_Files\' nome_exp_no_ '_01_nbasal_0001\' nome_exp_no_ '_01_nbasal_0001_' num2str(ch_indx) '.mat'])
        data_hpf=data;
        clear data
        
%         %% save data for cochlea
%         cd(parent_dir)
%         cd ..
%         all_for_cochlea=['all_for_nas\' name_exp '\ch' num2str(ch_indx)];
%         mkdir(all_for_cochlea)
%         save([all_for_cochlea '\data_raw_hpf'],'data_raw','data_hpf')
%         
        
        
        %% send to txt for R
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
        
        %% call r
        a=0;
        RunRcode('C:\Users\BuccelliLab\Documents\GitHub\burst_detection\Evaluation_code\use_methods.R','C:\Program Files\R\R-3.5.1\bin')
        
        %% move results from r in the right folder
        res_fold=['D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\results_from_r\' name_exp '\ch' num2str(ch_indx)];
        mkdir(res_fold)
        
        movefile("result*",res_fold)
        
%         %% load results from r session as tables
%         r.CMA = importfile([res_fold '\result_CMA.csv']);
%         r.Henning = importfile([res_fold '\result_hennig_method.csv']);
%         r.PS = importfile([res_fold '\result_PS_method.csv']);
%         r.RS = importfile([res_fold '\result_RS_method.csv']);
%         r.LogISI = importfile([res_fold '\result_pasquale_method.csv']);
%         r.MI = importfile([res_fold '\result_MI_method.csv']);
%         r.HSMM = importfile([res_fold '\result_HSMM_method.csv']);
%         
%         %% initialize signals for crosscorr
%         time_s=(1:1:length(peak_train))./fs;
%         s.cma=zeros(1,length(peak_train));
%         s.henning=zeros(1,length(peak_train));
%         s.ps=zeros(1,length(peak_train));
%         s.rs=zeros(1,length(peak_train));
%         s.logisi=zeros(1,length(peak_train));
%         s.mi=zeros(1,length(peak_train));
%         s.hsmm=zeros(1,length(peak_train));
%         s.ch=zeros(1,length(peak_train));
%         s.nas=zeros(1,length(peak_train));
%         s.vi=zeros(1,length(peak_train));
%         
%         %% load detection from cochlea 200ms FAKE
%         cochlea_res= import_results_cochlea(['D:\Capocaccia\New_training_test_sets\results_from_cochlea\classification_done_ch' num2str(12) 'raw.csv']);
%         shift=0.2;
%         [start_stop_cochlea_dur,start_stop_cochlea] = get_start_stop_cochlea(cochlea_res,shift);
%         
%         %% start stop visual inspection FAKE
%         load('D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\burst_start_stop_ch12.mat')
%         start_stop_vi_duration=visual_inspection(:,2)-visual_inspection(:,1);
%         
%         %% populate signals and plot if flag=1
%         plot_flag = 0;
%         s = populate_signals(time_stamps_s,r,s,plot_flag,signal_color,signal_names,time_s,burst_CH_s,start_stop_cochlea,visual_inspection);
%         save_flag = 0;
%         if plot_flag && save_flag
%             burst_spikes_overlap = gcf;
%             savefig(burst_spikes_overlap,'Figures\burst_spikes_overlap')
%         end
%         %% plot burst duration
%         burst_numb_dur = plot_burst_numb(r, signal_color,signal_names,burst_CH_s,start_stop_cochlea,visual_inspection,burst_CH_dur,start_stop_cochlea_dur,start_stop_vi_duration);
%         
%         %% concatenate all signals
%         all_signals=[s.vi; s.nas ;s.ch; s.hsmm; s.mi; s.logisi; s.rs; s.ps; s.henning; s.cma]';
%         n_signals=size(all_signals,2);
%         
%         %% get pearson imagesc corrcoef
%         pc_imagesc = pears(all_signals, signal_names);
%         
%         %% correlation all vs all
%         window=1e5; %10s
%         xc=zeros(n_signals,n_signals,window*2+1);
%         lags=zeros(n_signals,n_signals,window*2+1);
%         max_peak=zeros(n_signals,n_signals);
%         max_lag=zeros(n_signals,n_signals);
%         for curr_signal_2=1:n_signals
%             disp(curr_signal_2)
%             for curr_signal=1:n_signals
%                 [xc(curr_signal,curr_signal_2,:),lags(curr_signal,curr_signal_2,:)]=xcorr(all_signals(:,curr_signal),all_signals(:,curr_signal_2),window,'coef');
%                 max_peak(curr_signal,curr_signal_2)=max(xc(curr_signal,curr_signal_2,:));
%                 loc_max_peak=find(xc(curr_signal,curr_signal_2,:)==max_peak(curr_signal,curr_signal_2),1);
%                 max_lag(curr_signal,curr_signal_2)=lags(curr_signal,curr_signal_2,loc_max_peak)/fs;
%             end
%         end
%         
%         %% vi vs all plots
%         [cc_overlap,sub_cc]=plot_cc_vi_vs_all(xc,lags,signal_color,signal_names,fs);
%         
%         %% plot max peak and lags
%         cc_max_lag_imagesc = plot_max_peak_lag(max_peak,max_lag,signal_names);
%         
%         %% burst + raw + hpf
%         figure_burst_raw_hpf = plot_burst_raw_hpf(all_signals,peak_train,time_s,signal_color,data_raw,data_hpf);
        
    end
end

%%

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

function [s] = populate_signals(time_stamps_s,r,s,plot_flag,signal_color,signal_names,time_s,burst_CH_s,start_stop_cochlea,visual_inspection)
%% plot spikes and burst detection
if plot_flag
    burst_spikes_overlap=figure;
end
for curr_spike=1:length(time_stamps_s)
    curr_sample=time_stamps_s(curr_spike);
    if plot_flag
        plot([curr_sample curr_sample],[0 1.8],'Color',[.9 .9 .9],'LineWidth',0.2)
        hold on
    end
end
% plot bursts from r
for curr_burst=1:height(r.CMA)
    start_sample_s=time_stamps_s(r.CMA.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.CMA.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2),'Marker','o','MarkerEdgeColor',signal_color(1,:),'Color',signal_color(1,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.cma(start_sample:stop_sample)=1;
end
for curr_burst=1:height(r.Henning)
    start_sample_s=time_stamps_s(r.Henning.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.Henning.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+.2,'Marker','o','MarkerEdgeColor',signal_color(2,:),'Color',signal_color(2,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.henning(start_sample:stop_sample)=1;
end
for curr_burst=1:height(r.PS)
    start_sample_s=time_stamps_s(r.PS.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.PS.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+.4,'Marker','o','MarkerEdgeColor',signal_color(3,:),'Color',signal_color(3,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.ps(start_sample:stop_sample)=1;
end
for curr_burst=1:height(r.RS)
    start_sample_s=time_stamps_s(r.RS.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.RS.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+.6,'Marker','o','MarkerEdgeColor',signal_color(4,:),'Color',signal_color(4,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.rs(start_sample:stop_sample)=1;
end
% plot bursts from Paquale LogISI
for curr_burst=1:size(r.LogISI,1)
    start_sample_s=time_stamps_s(r.LogISI.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.LogISI.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+.8,'Marker','o','MarkerEdgeColor',signal_color(5,:),'Color',signal_color(5,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.logisi(start_sample:stop_sample)=1;
end
% plot bursts from MI
for curr_burst=1:size(r.MI,1)
    start_sample_s=time_stamps_s(r.MI.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.MI.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+1,'Marker','o','MarkerEdgeColor',signal_color(6,:),'Color',signal_color(6,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.mi(start_sample:stop_sample)=1;
end
% plot bursts from HSMM
for curr_burst=1:size(r.HSMM,1)
    start_sample_s=time_stamps_s(r.HSMM.beg(curr_burst));
    stop_sample_s=time_stamps_s(r.HSMM.end1(curr_burst));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+1.2,'Marker','o','MarkerEdgeColor',signal_color(7,:),'Color',signal_color(7,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.hsmm(start_sample:stop_sample)=1;
end
% plot bursts from matlab (Chiappalone)
for curr_burst=1:size(burst_CH_s,1)
    start_sample_s=time_stamps_s(time_stamps_s==burst_CH_s(curr_burst,1));
    stop_sample_s=time_stamps_s(time_stamps_s==burst_CH_s(curr_burst,2));
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+1.4,'Marker','o','MarkerEdgeColor',signal_color(8,:),'Color',signal_color(8,:))
        %
    end
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.ch(start_sample:stop_sample)=1;
end
% plot bursts from cochlea
for curr_burst=1:size(start_stop_cochlea,1)
    start_sample_s=start_stop_cochlea(curr_burst,1);
    stop_sample_s=start_stop_cochlea(curr_burst,2);
    if plot_flag
        %     plot([start_sample_s stop_sample_s],zeros(1,2)+1.6,'Marker','o','MarkerEdgeColor',signal_color(9,:),'Color',signal_color(9,:))
    end
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    s.nas(start_sample:stop_sample)=1;
end
% % get bursts from cochlea 50ms
% for curr_burst=1:size(start_stop_cochlea_50,1)
%     start_sample_s=start_stop_cochlea_50(curr_burst,1);
%     stop_sample_s=start_stop_cochlea_50(curr_burst,2);
%     plot([start_sample_s stop_sample_s],zeros(1,2)+1.6,'Marker','o','MarkerEdgeColor',signal_color(9,:),'Color',signal_color(9,:))
%     %
%     diff_start=abs(start_sample_s-time_s);
%     start_sample=find(diff_start==(min(diff_start)));
%     diff_stop=abs(stop_sample_s-time_s);
%     stop_sample=find(diff_stop==(min(diff_stop)));
%     s.nas_50(start_sample:stop_sample)=1;
% end
% plot bursts from visual inspection
for curr_burst=1:size(visual_inspection,1)
    start_sample_s=visual_inspection(curr_burst,1);
    stop_sample_s=visual_inspection(curr_burst,2);
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+1.8,'Marker','o','MarkerEdgeColor',signal_color(10,:),'Color',signal_color(10,:))
        %
    end
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    s.vi(start_sample:stop_sample)=1;
end

if plot_flag
    % improve graphics
    ylim([-.1 1.9])
    title('spike detection comparison')
    xlabel('Time [s]')
    yticks([0 .2 .4 .6 .8 1 1.2 1.4 1.6 1.8])
    yticklabels(fliplr(signal_names))
end

end

function burst_numb_dur = plot_burst_numb(r, signal_color, signal_names,burst_CH_s,start_stop_cochlea,visual_inspection,burst_CH_dur,start_stop_cochlea_dur,start_stop_vi_duration)
%% comparing number of bursts detected
burst_numb_dur=figure;
subplot(3,1,1)
n_bursts=[height(r.CMA) height(r.Henning) height(r.PS) height(r.RS) height(r.LogISI) height(r.MI) height(r.HSMM) size(burst_CH_s,1) size(start_stop_cochlea,1) size(visual_inspection,1)];
b=bar(n_bursts(1:end-1));
b.FaceColor = 'flat';
b.CData = signal_color(1:end-1,:);
hold on
plot([0 10],[size(visual_inspection,1) size(visual_inspection,1)],'k--')
xticklabels(fliplr(signal_names(2:end)))
ylabel('# of bursts')
title('comparing number of bursts detected 200ms')

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
mean_vals=[mean(r.CMA.durn) mean(r.Henning.durn) mean(r.PS.durn) mean(r.RS.durn) mean(r.LogISI.durn) mean(r.MI.durn) mean(r.HSMM.durn) mean(burst_CH_dur) mean(start_stop_cochlea_dur) mean(start_stop_vi_duration)];
std_vals=[std(r.CMA.durn) std(r.Henning.durn) std(r.PS.durn) std(r.RS.durn) std(r.LogISI.durn) std(r.MI.durn) std(r.HSMM.durn) std(burst_CH_dur) std(start_stop_cochlea_dur) std(start_stop_vi_duration)];
se_vals=std_vals./sqrt(n_bursts);

subplot(3,1,3)
b=bar(mean_vals);
b.FaceColor = 'flat';
b.CData = signal_color;
hold on
errorbar(mean_vals,se_vals,'.k')
xticklabels(fliplr(signal_names))
ylabel('Mean burst duration (s)')
title('comparing mean burst duration +-SE')
end

function pc_imagesc = pears(all_signals, signal_names)
corr_matrix=corrcoef(all_signals);
pc_imagesc=figure;
imagesc(corr_matrix)
yticklabels(signal_names)
xticklabels(signal_names)
colorbar
title('Pearson coefficient')
end

function [cc_overlap,sub_cc]=plot_cc_vi_vs_all(xc,lags,signal_color,signal_names,fs)
%% correlation VI vs all
cc_overlap=figure;
for curr_signal=1:10
    plot(squeeze(lags(1,curr_signal,:))/fs,squeeze(xc(1,curr_signal,:)),'Color',signal_color(11-curr_signal,:))
    hold on
end
legend(signal_names)
title('cross correlation between VI and the other methods')
xlabel('Time [s]')

%% subplots
sub_cc=figure;
for curr_signal=1:10
    h(curr_signal)=subplot(2,5,curr_signal);
    plot(squeeze(lags(1,curr_signal,:))/fs,squeeze(xc(1,curr_signal,:)),'LineWidth',3,'Color',signal_color(11-curr_signal,:))
    max_peak=max(xc(1,curr_signal,:));
    loc_max_peak=find(xc(1,curr_signal,:)==max_peak,1);
    max_lag=lags(1,curr_signal,loc_max_peak)/fs;
    title([signal_names{curr_signal} ', p:' num2str(max_peak,'%.2f') ', L:' num2str(max_lag,'%.2f')])
    xlabel('Time [s]')
end
linkaxes(h,'xy')
end

function cc_max_lag_imagesc = plot_max_peak_lag(max_peak,max_lag,signal_names)
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
% savefig(cc_max_lag_imagesc,'Figures\cc_max_lag_imagesc')
end

function figure_burst_raw_hpf = plot_burst_raw_hpf(all_signals,peak_train,time_s,signal_color,data_raw,data_hpf)

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
end