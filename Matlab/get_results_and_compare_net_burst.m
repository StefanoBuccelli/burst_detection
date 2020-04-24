%% get results from r
% all_figures\20180226_27557\burst_raw_ch52.fig NAS better (raw more
% informative)


clear
close all
clc
fs=1e4;

parent_dir='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all\';
parent_dir_raw='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_raw\';
cd(parent_dir)
fig_dir='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_figures\';
load('train_tab.mat')
cd ..
signal_names=fliplr({'CMA','ISIrank','PS','RS','LogISI','MI','HSMM','CH','NAS','VI','NB'});
n_signals=length(signal_names);
n_test_channels=14;
signal_color=jet(10);
%% load one channel spike detection
n_bursts = zeros(n_test_channels,n_signals);
mean_vals= zeros(n_test_channels,n_signals);
se_vals= zeros(n_test_channels,n_signals);
std_vals= zeros(n_test_channels,n_signals);
n_bursts_ratio = zeros(n_test_channels,n_signals);
mean_vals_ratio= zeros(n_test_channels,n_signals);
corr_matrix=zeros(n_signals,n_signals,n_test_channels);
max_peak=zeros(n_signals,n_signals,n_test_channels);
max_lag=zeros(n_signals,n_signals,n_test_channels);
window=1e5; %10s
xc=zeros(n_signals,n_signals,window*2+1,n_test_channels);
plot_flag = 0; % plot and save

        
rec_count=1;
for curr_ch = 1:2
    for curr_exp=1 : height(train_tab)
        disp(rec_count)
        name_exp=train_tab.Properties.RowNames{curr_exp};
        if curr_ch==1
            ch_indx=train_tab.Ch1(curr_exp);
        else
            ch_indx=train_tab.Ch2(curr_exp);
        end
        curr_dir_peak_train=[name_exp '_HFP300_3000\' name_exp '_PeakDetectionMAT_PLP2ms_RP1ms\ptrain_' name_exp '_01_nbasal_0001'];
        load([parent_dir curr_dir_peak_train '\ptrain_' name_exp '_01_nbasal_0001_' num2str(ch_indx) '.mat'])
        time_stamps_samples=find(peak_train);
        time_stamps_s=time_stamps_samples/fs;
        
        if plot_flag
            mkdir([fig_dir name_exp])
        end
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
        
        %% move results from r in the right folder
        res_fold_r=['D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\results_from_r\' name_exp '\ch' num2str(ch_indx)];
        
        %% load results from r session as tables
        r.CMA = importfile([res_fold_r '\result_CMA.csv']);
        r.Henning = importfile([res_fold_r '\result_hennig_method.csv']);
        r.PS = importfile([res_fold_r '\result_PS_method.csv']);
        r.RS = importfile([res_fold_r '\result_RS_method.csv']);
        r.LogISI = importfile([res_fold_r '\result_pasquale_method.csv']);
        r.MI = importfile([res_fold_r '\result_MI_method.csv']);
        r.HSMM = importfile([res_fold_r '\result_HSMM_method.csv']);
%         
        %% initialize signals for crosscorr
        time_s=(1:1:length(peak_train))./fs;
        s.cma=zeros(1,length(peak_train));
        s.henning=zeros(1,length(peak_train));
        s.ps=zeros(1,length(peak_train));
        s.rs=zeros(1,length(peak_train));
        s.logisi=zeros(1,length(peak_train));
        s.mi=zeros(1,length(peak_train));
        s.hsmm=zeros(1,length(peak_train));
        s.ch=zeros(1,length(peak_train));
        s.nas=zeros(1,length(peak_train));
        s.vi=zeros(1,length(peak_train));
        s.nb=zeros(1,length(peak_train));

        res_fold_nas=['D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_from_nas\' name_exp ];
        
%         mkdir(res_fold_nas)
        cd(res_fold_nas)
%         cd ..
%         file_=dir([pwd '\*ch' num2str(ch_indx) '*']);
%         if ~isempty(file_)
%             movefile([pwd '\*ch' num2str(ch_indx) '*'],res_fold_nas)
%         end


        %% load detection from cochlea 50ms 
        res_nas_ch=[res_fold_nas '\classification_done_ch' num2str(ch_indx) 'raw_50ms.csv'];
        cochlea_res= import_results_cochlea(res_nas_ch);
        shift=0.05;
        [start_stop_cochlea_dur,start_stop_cochlea] = get_start_stop_cochlea(cochlea_res,shift);
%         
        %% start stop visual inspection 
%         load('D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\burst_start_stop_ch12.mat')
%         start_stop_vi_duration=visual_inspection(:,2)-visual_inspection(:,1);
        dir_from_vi='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_from_VI\';
        load([dir_from_vi name_exp '\' name_exp '_burststartstop_' num2str(ch_indx) '.mat'])
        visual_inspection=burst_start_stop;
        clear burst_start_stop
        start_stop_vi_duration=visual_inspection(:,2)-visual_inspection(:,1);   
        
        %% start stop network burst
        net_burst_dir='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\nb\';
        load([net_burst_dir name_exp '_BurstDetectionMAT_5-100msec\' name_exp(1:end-5) 'NetworkBurstDetectionFiles\' name_exp '_BurstDetectionFiles_NetworkBurstDetection_' name_exp '_01_nbasal_0001.mat.mat'])
        %limit to first 300s
        netBursts=netBursts./fs;
        start_stop_NB_duration=netBursts(:,2)-netBursts(:,1);
        
        %% populate signals and plot if flag=1
        
        s = populate_signals(time_stamps_s,r,s,plot_flag,signal_color,signal_names,time_s,burst_CH_s,start_stop_cochlea,visual_inspection,netBursts);
        
        if plot_flag
            burst_spikes_overlap = gcf;
            savefig(burst_spikes_overlap,[fig_dir name_exp '\burst_spikes_overlap_ch' num2str(ch_indx)])
        end
        %% get burst number and duration
        n_bursts(rec_count,:)=[height(r.CMA) height(r.Henning) height(r.PS) height(r.RS) height(r.LogISI) height(r.MI) height(r.HSMM) size(burst_CH_s,1) size(start_stop_cochlea,1) size(visual_inspection,1) size(netBursts,1)];
        mean_vals(rec_count,:)=[mean(r.CMA.durn) mean(r.Henning.durn) mean(r.PS.durn) mean(r.RS.durn) mean(r.LogISI.durn) mean(r.MI.durn) mean(r.HSMM.durn) mean(burst_CH_dur) mean(start_stop_cochlea_dur) mean(start_stop_vi_duration) mean(start_stop_NB_duration)];
        std_vals(rec_count,:)=[std(r.CMA.durn) std(r.Henning.durn) std(r.PS.durn) std(r.RS.durn) std(r.LogISI.durn) std(r.MI.durn) std(r.HSMM.durn) std(burst_CH_dur) std(start_stop_cochlea_dur) std(start_stop_vi_duration) std(start_stop_NB_duration)];
        se_vals(rec_count,:)=std_vals(rec_count,:)./sqrt(n_bursts(rec_count,:));
        n_bursts_ratio(rec_count,:)=n_bursts(rec_count,:)./n_bursts(rec_count,10);
        mean_vals_ratio(rec_count,:)=mean_vals(rec_count,:)./mean_vals(rec_count,10);
        %% plot burst duration
        if plot_flag
            burst_numb_dur = plot_burst_numb(signal_color,signal_names,visual_inspection, n_bursts(rec_count,:), mean_vals(rec_count,:), se_vals(rec_count,:),name_exp, ch_indx);
            savefig(burst_numb_dur,[fig_dir name_exp '\burst_numb_dur_ch' num2str(ch_indx)])
        end
%         
        %% concatenate all signals
        all_signals=[s.nb; s.vi; s.nas ;s.ch; s.hsmm; s.mi; s.logisi; s.rs; s.ps; s.henning; s.cma]';

        %% get pearson corrcoef
        corr_matrix(:,:,rec_count)=corrcoef(all_signals);
        
        %% plot imagesc corrcoef
        if plot_flag
            pc_imagesc = pears(corr_matrix(:,:,rec_count), signal_names,name_exp, ch_indx);
            savefig(pc_imagesc,[fig_dir name_exp '\pc_imagesc_ch' num2str(ch_indx)])
        end
%         
        %% correlation all vs all
        [xc(:,:,:,rec_count), lags, max_peak(:,:,rec_count), max_lag(:,:,rec_count) ] = get_corr(n_signals, all_signals,window, fs);
        curr_xc=squeeze(xc(:,:,:,rec_count));
        
        %% vi vs all plots
        if plot_flag
        [cc_overlap,sub_cc]=plot_cc_vi_vs_all(curr_xc,lags,signal_color,signal_names,fs);
        savefig(cc_overlap,[fig_dir name_exp '\cc_overlap_ch' num2str(ch_indx)])
        savefig(sub_cc,[fig_dir name_exp '\sub_cc_ch' num2str(ch_indx)])
        end
%         
        %% plot max peak and lags
        if plot_flag
            cc_max_lag_imagesc = plot_max_peak_lag(max_peak(:,:,rec_count),max_lag(:,:,rec_count),signal_names);
            savefig(cc_max_lag_imagesc,[fig_dir name_exp '\cc_max_lag_imagesc_ch' num2str(ch_indx)])
        end
        %         %% burst + raw + hpf
        if plot_flag
            figure_burst_raw_hpf = plot_burst_raw_hpf(all_signals,peak_train,time_s,signal_color,data_raw,data_hpf);
            savefig(figure_burst_raw_hpf,[fig_dir name_exp '\burst_raw_ch' num2str(ch_indx)])
        end
        close all
        rec_count=rec_count+1;
    end
end

%% burst number all
figure;
plot(n_bursts','Color',[.9 .9 .9])
hold on
boxplot(n_bursts)
xticklabels(fliplr(signal_names))
title('number of burst')
ylabel('burst numb')
% savefig(burst_ratio_all,[fig_dir 'burst_ratio_all'])

%% burst number all
burst_ratio_all=figure;
boxplot(n_bursts_ratio)
xticklabels(fliplr(signal_names))
title('number of burst ratio vs VI')
ylabel('burst ratio')
% savefig(burst_ratio_all,[fig_dir 'burst_ratio_all'])

%% burst duration all
burst_duration_all=figure;
boxplot(mean_vals_ratio)
xticklabels(fliplr(signal_names))
title('burst duration ratio vs VI')
ylabel('burst duration ratio')
% savefig(burst_duration_all,[fig_dir 'burst_duration_all'])

%% pearson all
pearson_all=figure;
imagesc(mean(corr_matrix,3))
yticklabels(signal_names)
xticklabels(signal_names)
title('mean pearson coeff all')
% savefig(pearson_all,[fig_dir 'pearson_all'])

%% mean max corr
max_peak_lag_all=figure;
subplot(1,2,1)
imagesc(mean(max_peak,3))
xticks(1:n_signals)
yticks(1:n_signals)
yticklabels(signal_names)
xticklabels(signal_names)
title('mean max corr all')
subplot(1,2,2)
imagesc(mean(max_lag,3))
xticks(1:n_signals)
yticks(1:n_signals)
yticklabels(signal_names)
xticklabels(signal_names)
title('mean lag all')
% savefig(max_peak_lag_all,[fig_dir 'max_peak_lag_all'])

%% VI vs all all max peaks and lags
boxp_max_peak_lag_all=figure;
subplot(2,1,1)
boxplot(squeeze(max_peak(1,:,:))')
xticks(1:n_signals)
xticklabels(signal_names)
title('max peak vs NB')
subplot(2,1,2)
boxplot(squeeze(max_lag(1,:,:))')
xticks(1:n_signals)
xticklabels(signal_names)
title('lag at max peak vs NB')
% savefig(boxp_max_peak_lag_all,[fig_dir 'boxp_max_peak_lag_all'])

%% VI vs all cc function
for curr_method=1:n_signals
cc_method_ch_subplot=figure;
rec_count=1;
for curr_ch = 1:2
    for curr_exp=1 : height(train_tab)
        name_exp=train_tab.Properties.RowNames{curr_exp};
        if curr_ch==1
            ch_indx=train_tab.Ch1(curr_exp);
        else
            ch_indx=train_tab.Ch2(curr_exp);
        end
        h_all(rec_count)=subplot(2,7,rec_count);
        plot(squeeze(lags(1,1,:))./fs,squeeze(xc(1,2,:,rec_count)),'b')
        hold on
        plot(squeeze(lags(1,1,:))./fs,squeeze(xc(1,curr_method,:,rec_count)),'r')
        title([name_exp ' ch' num2str(ch_indx)],'Interpreter','none')
        rec_count=rec_count+1;
    end
end
linkaxes(h_all,'xy')
xlim([-2 2])
savefig(cc_method_ch_subplot,[fig_dir 'cc_nas_' signal_names{curr_method} '_subplot'])
close
end

%% count how many times (maximum 14) NAS was better than the others
curr_great_than_nas=zeros(n_signals-2,1);
vi_method_all=[];
for curr_m_1=1:8
    vi_nas=squeeze(max_peak(1,2,:));
    vi_method=squeeze(max_peak(curr_m_1+2,1,:));
    curr_great_than_nas(curr_m_1,1)=sum(vi_nas>=vi_method);
    vi_method_all=[vi_method_all vi_method];
end
max_peak_perc_box_bar=figure;
subplot(2,1,2)
bar(100*curr_great_than_nas./14)
xticks(1:8)
xticklabels(signal_names(3:end))
ylim([0 110])
title('perc of chs with a peak NAS-VI greater than method-VI')

% all ratios
subplot(2,1,1)
plot([vi_method_all./vi_nas]','Color',[.9 .9 .9],'Marker','.')
hold on
boxplot([vi_method_all./vi_nas])
xticks(1:8)
xticklabels(signal_names(3:end))
title('max peak per ch per method (VI vs all) / NAS')
savefig(max_peak_perc_box_bar,[fig_dir 'max_peak_perc_box_bar'])


%% all raw
max_peak_per_ch_boxplot=figure;
plot([vi_nas vi_method_all]','Color',[.9 .9 .9],'Marker','.')
hold on
boxplot([vi_nas vi_method_all])
xticks(1:9)
xticklabels(signal_names(2:end))
title('max peak per ch per method')
savefig(max_peak_per_ch_boxplot,[fig_dir 'max_peak_per_ch_boxplot'])
%%
% save([fig_dir 'all_data'])

%% count how many times (maximum 14) NAS was better than the others
curr_great_than_logisi=zeros(n_signals,1);
logisi_method_all=[];
for curr_m_1=1:10
    logisi_nas=squeeze(max_peak(2,6,:));
    logisi_method=squeeze(max_peak(curr_m_1,6,:));
    curr_great_than_logisi(curr_m_1,1)=sum(logisi_nas>=logisi_method);
    logisi_method_all=[logisi_method_all logisi_method];
end
max_peak_perc_box_bar=figure;
subplot(2,1,2)
bar(100*curr_great_than_logisi./14)
xticks(1:10)
xticklabels(signal_names(1:end))
ylim([0 110])
title('perc of chs with a peak NAS-logisi greater than method-logisi')

% all ratios
subplot(2,1,1)
plot([logisi_method_all./logisi_nas]','Color',[.9 .9 .9],'Marker','.')
hold on
boxplot([logisi_method_all./logisi_nas])
xticks(1:10)
xticklabels(signal_names(1:end))
title('max peak per ch per method (logisi vs all) / logisi-NAS')


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

function [s] = populate_signals(time_stamps_s,r,s,plot_flag,signal_color,signal_names,time_s,burst_CH_s,start_stop_cochlea,visual_inspection,netBursts)
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
        plot([start_sample_s stop_sample_s],zeros(1,2)+1.6,'Marker','o','MarkerEdgeColor',signal_color(9,:),'Color',signal_color(9,:))
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


for curr_burst=1:size(netBursts,1)
    start_sample_s=netBursts(curr_burst,1);
    stop_sample_s=netBursts(curr_burst,2);
    if plot_flag
        plot([start_sample_s stop_sample_s],zeros(1,2)+2,'Marker','o','MarkerEdgeColor',signal_color(10,:),'Color',signal_color(10,:))
        %
    end
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    s.nb(start_sample:stop_sample)=1;
end

if plot_flag
    % improve graphics
    ylim([-.1 2.1])
    title('spike detection comparison')
    xlabel('Time [s]')
    yticks([0 .2 .4 .6 .8 1 1.2 1.4 1.6 1.8 2])
    yticklabels(fliplr(signal_names))
end

end

function [burst_numb_dur] = plot_burst_numb(signal_color, signal_names,visual_inspection,n_bursts, mean_vals, se_vals, name_exp, ch_indx)
%% comparing number of bursts detected
burst_numb_dur=figure;
subplot(3,1,1)
b=bar(n_bursts(1:end-1));
b.FaceColor = 'flat';
b.CData = signal_color(1:end-1,:);
hold on
plot([0 10],[size(visual_inspection,1) size(visual_inspection,1)],'k--')
xticklabels(fliplr(signal_names(2:end)))
ylabel('# of bursts')
title([name_exp ' ch ' num2str(ch_indx) ': n of bursts'],'Interpreter','none')

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

function [pc_imagesc,corr_matrix] = pears(corr_matrix, signal_names,name_exp, ch_indx)
pc_imagesc=figure;
imagesc(corr_matrix)
yticklabels(signal_names)
xticklabels(signal_names)
colorbar
title([name_exp ' ch: ' num2str(ch_indx) ', Pearson coefficient'],'Interpreter','none')
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
yticklabels({'Spikes','CMA','ISIrank','PS','RS','LogISI','MI','HSMM','CH','NAS','VI'})
xticks([])
h_b(2)=subplot(9,1,8);
plot(time_s,data_raw,'Color','k')
xticks([])
h_b(3)=subplot(9,1,9);
plot(time_s,data_hpf,'Color','k')
xlabel('Time [s]')
linkaxes(h_b,'x')
end

function [xc, lags, max_peak, max_lag ] = get_corr(n_signals, all_signals, window, fs)
xc=zeros(n_signals,n_signals,window*2+1);
lags=zeros(n_signals,n_signals,window*2+1);
max_peak=zeros(n_signals,n_signals);
max_lag=zeros(n_signals,n_signals);
for curr_signal_2=1:n_signals
    disp(curr_signal_2)
    for curr_signal=1:n_signals
        [xc(curr_signal,curr_signal_2,:),lags(curr_signal,curr_signal_2,:)]=xcorr(all_signals(:,curr_signal),all_signals(:,curr_signal_2),window,'coef');
        max_peak(curr_signal,curr_signal_2)=max(xc(curr_signal,curr_signal_2,:));
        loc_max_peak=find(xc(curr_signal,curr_signal_2,:)==max_peak(curr_signal,curr_signal_2),1);
        max_lag(curr_signal,curr_signal_2)=lags(curr_signal,curr_signal_2,loc_max_peak)/fs;
    end
end
end