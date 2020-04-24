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
load('D:\Capocaccia\New_training_test_sets\Training_set\training_data_ch12_both_raw_hpf.mat')

%%
% figure
% [n,edges]=histcounts(raw_burst_concat_linear,'Normalization','cdf');
% plot(edges(1,1:end-1),n)
% hold on
% [n_no,edges_no]=histcounts(raw_no_burst_concat_linear,'Normalization','cdf');
% plot(edges_no(1,1:end-1),n_no)

%% Roc sample by sample
% roc_data=roc_curve(raw_no_burst_concat_linear',raw_burst_concat_linear);

%% binning 50ms
win=500;
%% burst
train.data_burst_resh=reshape(raw_burst_concat_linear(1:win*floor(length(raw_burst_concat_linear)/win)),win,[]);
train.data_burst_resh_max=max(train.data_burst_resh);
train.data_burst_resh_min=max(-train.data_burst_resh);
train.data_burst_resh_len=sum(abs(diff((train.data_burst_resh))));

%% no burst
train.data_no_burst_resh=reshape(raw_no_burst_concat_linear(1:win*floor(length(raw_no_burst_concat_linear)/win)),win,[]);
train.data_no_burst_resh_max=max(train.data_no_burst_resh);
train.data_no_burst_resh_min=max(-train.data_no_burst_resh);
train.data_no_burst_resh_len=sum(abs(diff(train.data_no_burst_resh)));

%% roc on max values per 50ms
figure
roc.data_max=roc_curve(train.data_no_burst_resh_max',train.data_burst_resh_max');
figure
roc.data_len=roc_curve(train.data_no_burst_resh_len',train.data_burst_resh_len');
figure
roc.data_min=roc_curve(train.data_no_burst_resh_min',train.data_burst_resh_min');

%% histogram non burst burst
figure
histogram(train.data_no_burst_resh_max,'Normalization','probability')
hold on
histogram(train.data_burst_resh_max,'Normalization','probability','BinWidth',1)
plot([roc.data_max.param.Threshold roc.data_max.param.Threshold],[0 0.2],'LineWidth',3)
title('Training set max per 50ms')
legend('No burst','Burst','Best threshold')

%% histogram non burst burst
figure
histogram(train.data_no_burst_resh_min,'Normalization','probability')
hold on
histogram(train.data_burst_resh_min,'Normalization','probability','BinWidth',1)
plot([roc.data_min.param.Threshold roc.data_min.param.Threshold],[0 0.2],'LineWidth',3)
title('Training set min per 50ms')
legend('No burst','Burst','Best threshold')

%% histogram non burst burst
figure
histogram(train.data_no_burst_resh_len,'Normalization','probability')
hold on
histogram(train.data_burst_resh_len,'Normalization','probability','BinWidth',10)
plot([roc.data_len.param.Threshold roc.data_len.param.Threshold],[0 0.2],'LineWidth',3)
title('Training set signal len per 50ms')
legend('No burst','Burst','Best threshold')

save('train_roc','train','roc')
%%
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
        
        %%
        th.max=roc.data_max.param.Threshold;
        th.len=roc.data_len.param.Threshold;
        
        %% reshape
        data_raw_reshape=reshape(data_raw,win,[]);
        
        shift=win/fs;
        time_s=(1:1:length(peak_train))./fs;
        %% max
        data_raw_resh.max=max(data_raw_reshape);
        loc_above_th_max=find(data_raw_resh.max>th.max);
        vals_above_th_max=data_raw_resh.max(data_raw_resh.max>th.max);
        
        raw_res_max.class=zeros(1,size(data_raw_reshape,2));
        raw_res_max.class(1,loc_above_th_max)=1;
        raw_res_max.Time=(1:size(data_raw_reshape,2))*win/fs;
        
        [start_stop_raw_dur.max,start_stop_raw.max] = get_start_stop_raw(raw_res_max,shift);
        %% len
        data_raw_resh.len=sum(abs(diff(data_raw_reshape)));
        loc_above_th_len=find(data_raw_resh.len>th.len);
        vals_above_th_len=data_raw_resh.max(data_raw_resh.len>th.len);
        
        raw_res_len.class=zeros(1,size(data_raw_reshape,2));
        raw_res_len.class(1,loc_above_th_len)=1;
        raw_res_len.Time=(1:size(data_raw_reshape,2))*win/fs;
        
        [start_stop_raw_dur.len,start_stop_raw.len] = get_start_stop_raw(raw_res_len,shift);
        %% populate
        raw_methods={'max','len'};
        s_raw = populate_s_raw(raw_methods,peak_train,start_stop_raw,time_s);
        
        %% figures
        param_per_win='signal length';
        plot_raw_histogram=zeros(size(raw_methods));
        for curr_method=1:length(raw_methods)
            method_name=raw_methods{curr_method};
            plot_raw_histogram(curr_method)=plot_raw_hist(data_raw,s_raw,method_name,data_raw_resh,name_exp,th);
        end
    end
end

function [plot_raw_histogram]=plot_raw_hist(data_raw,s_raw,method_name,data_raw_resh,name_exp,th)
plot_raw_histogram=figure;
h(1)=subplot(3,1,1);
plot(data_raw)
title([name_exp ' ' method_name],'Interpreter','none')
h(2)=subplot(3,1,2);
plot(s_raw.(method_name))
ylim([0 1.1])
subplot(3,1,3)
histogram(data_raw_resh.(method_name),'Normalization','probability')
hold on
plot([th.(method_name) th.(method_name)],[0 0.1],'r')
legend({method_name,'threshold'})
title(['histogram 50ms ' method_name])
linkaxes(h(1:2),'x')
end

function [start_stop_raw_dur,start_stop_raw] = get_start_stop_raw(raw_res,shift)

raw_res.Time=raw_res.Time-raw_res.Time(1); %restart from the first normal value

raw_res.Time_stretch=linspace(0,299.9,length(raw_res.Time))'; %% I removed 200ms which is roughtly the duration of a window

%%
falling_edge=1; %% initialize to find rising edge first
rising_edge=0;
new_rising=0;
start_stop_raw=[];
for curr_time=2:length(raw_res.Time)
    % if falling_edge==1 look for rising edge
    if falling_edge==1
        if (raw_res.class(curr_time)-raw_res.class(curr_time-1))==1
            rising_edge=1;
            falling_edge=0;
            new_rising=new_rising+1;
            start_stop_raw(new_rising,1)=raw_res.Time_stretch(curr_time)-shift;%subtract 100ms from start assuming original timing is the central
            %            start_stop_cochlea(new_rising,1)=cochlea_res.Time_stretch(curr_time);
        else
            rising_edge=0;
        end
    else  %% look for falling edge
        if (raw_res.class(curr_time)-raw_res.class(curr_time-1))==(-1)
            falling_edge=1;
            start_stop_raw(new_rising,2)=raw_res.Time_stretch(curr_time)-shift;%subtract 100ms from end assuming original timing is the central
            %            start_stop_cochlea(new_rising,2)=cochlea_res.Time_stretch(curr_time);
        else
            falling_edge=0;
        end
    end
end
start_stop_raw_dur=start_stop_raw(:,2)-start_stop_raw(:,1);
end

function [s_raw] = populate_s_raw(raw_methods,peak_train,start_stop_raw,time_s)
for curr_method=1:length(raw_methods)
    method_name=raw_methods{curr_method};
    s_raw.(method_name)=zeros(1,length(peak_train));
    for curr_burst=1:size(start_stop_raw.(method_name),1)
        start_sample_s=start_stop_raw.(method_name)(curr_burst,1);
        stop_sample_s=start_stop_raw.(method_name)(curr_burst,2);
        diff_start=abs(start_sample_s-time_s);
        start_sample=find(diff_start==(min(diff_start)));
        diff_stop=abs(stop_sample_s-time_s);
        stop_sample=find(diff_stop==(min(diff_stop)));
        s_raw.(method_name)(start_sample:stop_sample)=1;
    end
end
end