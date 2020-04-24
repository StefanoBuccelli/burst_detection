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
fig_dir_raw=[fig_dir 'fig_raw_methods\'];
mkdir(fig_dir_raw)
load('train_tab.mat')
cd ..
load('D:\Capocaccia\New_training_test_sets\Training_set\training_data_ch12_both_raw_hpf.mat')


%% raw methods
raw_methods={'max','peak_peak','len'};
other_methods={'VI','NAS','CH'};

all_methods = ['VI','NAS', raw_methods]; %% no ch in this case
n_test_channels=14;

%%
n_signals=length(all_methods);
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
train.data_burst.max=max(train.data_burst_resh);
train.data_burst.min=max(-train.data_burst_resh);
train.data_burst.len=sum(abs(diff((train.data_burst_resh))));
train.data_burst.peak_peak=max(train.data_burst_resh)-min(train.data_burst_resh);

%% no burst
train.data_no_burst_resh=reshape(raw_no_burst_concat_linear(1:win*floor(length(raw_no_burst_concat_linear)/win)),win,[]);
train.data_no_burst.max=max(train.data_no_burst_resh);
train.data_no_burst.min=max(-train.data_no_burst_resh);
train.data_no_burst.len=sum(abs(diff(train.data_no_burst_resh)));
train.data_no_burst.peak_peak=max(train.data_no_burst_resh)-min(train.data_no_burst_resh);

%% roc on max values per 50ms
h_roc=figure;
for curr_method=1:length(raw_methods)
    subplot(2,length(raw_methods),curr_method)
    method_name=raw_methods{curr_method};
    roc.(method_name)=roc_curve(train.data_no_burst.(method_name)',train.data_burst.(method_name)');
    h1=gca;
    auc=h1.Title.String;
    title(['method: ' method_name '; ' auc],'Interpreter','none')
end
for curr_method=1:length(raw_methods)
        method_name=raw_methods{curr_method};
        subplot(2,length(raw_methods),curr_method+length(raw_methods))
        [n_nob,eng_nob]=histcounts(train.data_no_burst.(method_name),'Normalization','probability');
        plot(eng_nob(1:end-1)+diff(eng_nob)/2,n_nob)
        hold on
        [n_b,eng_b]=histcounts(train.data_burst.(method_name),'Normalization','probability');
        bin_w_old=diff(eng_b(1:2));
        bin_w_new=bin_w_old/4;
        [n_b,eng_b]=histcounts(train.data_burst.(method_name),'Normalization','probability','BinWidth',bin_w_new);
        plot(eng_b(1:end-1)+diff(eng_b)/2,smooth(n_b,4))
%         plot(eng_b(1:end-1)+diff(eng_b)/2,(n_b))
        plot([roc.(method_name).param.Threshold roc.(method_name).param.Threshold],[0 0.2],'LineWidth',2)
        xlabel(method_name,'Interpreter','none')
        title(['Training set per 50ms, method: ' method_name],'Interpreter','none')
        legend('No burst','Burst','Best threshold')
end

%% extract thresholds
for curr_method=1:length(raw_methods)
    method_name=raw_methods{curr_method};
    th.(method_name)=roc.(method_name).param.Threshold;
end

%% save train and roc
save('train_roc','train','roc')
%%
roc_test_all=cell(2*height(train_tab),1);
perc_NAS_VI=zeros(2*height(train_tab),1);
perc_CH_VI=zeros(2*height(train_tab),1);
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
        
        
        %% load detection from cochlea 50ms
        res_fold_nas=['D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_from_nas\' name_exp ];
        cd(res_fold_nas)
        res_nas_ch=[res_fold_nas '\classification_done_ch' num2str(ch_indx) 'raw_50ms.csv'];
        cochlea_res= import_results_cochlea(res_nas_ch);
        shift=0.05;
        [start_stop_cochlea_dur,start_stop_cochlea] = get_start_stop_cochlea(cochlea_res,shift);
%         
        %% start stop visual inspection FAKE
%         load('D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\burst_start_stop_ch12.mat')
%         start_stop_vi_duration=visual_inspection(:,2)-visual_inspection(:,1);
        dir_from_vi='D:\OneDrive - Fondazione Istituto Italiano Tecnologia\Capocaccia_burst_detection\Test_set\all_from_VI_2\';
        load([dir_from_vi name_exp '\' name_exp '_burststartstop_' num2str(ch_indx) '.mat'])
        visual_inspection=burst_start_stop;
        clear burst_start_stop
        start_stop_vi_duration=visual_inspection(:,2)-visual_inspection(:,1);  
        
        %% load raw and hpf
        nome_exp_no_=name_exp;
        nome_exp_no_(9)=[];
        load([parent_dir_raw nome_exp_no_ '\' nome_exp_no_ '_Mat_files\' nome_exp_no_ '_01_nbasal_0001\' nome_exp_no_ '_01_nbasal_0001_' num2str(ch_indx) '.mat'])
        data_raw=data;
        clear data
        load([parent_dir_raw nome_exp_no_ '\' nome_exp_no_ '_FilteredData\' nome_exp_no_ '_Mat_Files\' nome_exp_no_ '_01_nbasal_0001\' nome_exp_no_ '_01_nbasal_0001_' num2str(ch_indx) '.mat'])
        data_hpf=data;
        clear data
        
        
        %% reshape raw
        data_raw_reshape=reshape(data_raw,win,[]);
        
        shift=win/fs;
        time_s=(1:1:length(peak_train))./fs;
        
        %% add here if need more methods
        data_raw_resh.max=max(data_raw_reshape);
        data_raw_resh.len=sum(abs(diff(data_raw_reshape)));
        data_raw_resh.peak_peak=max(data_raw_reshape)-min(data_raw_reshape);
        
        %% for each method get burs start stop and duration
        for curr_method=1:length(raw_methods)
            method_name=raw_methods{curr_method};
            loc_above_th.(method_name)=find(data_raw_resh.(method_name)>th.(method_name));
            vals_above_th.(method_name)=data_raw_resh.(method_name)(data_raw_resh.(method_name)>th.(method_name));
            
            raw_res.(method_name).class=zeros(1,size(data_raw_reshape,2));
            raw_res.(method_name).class(1,loc_above_th.(method_name))=1;
            raw_res.(method_name).Time=(1:size(data_raw_reshape,2))*win/fs;
            
            [start_stop_raw_dur.(method_name),start_stop_raw.(method_name)] = get_start_stop_raw(raw_res.(method_name),shift);
        end

        %% populate raw
        s_raw = populate_s_raw(raw_methods,peak_train,start_stop_raw,time_s,win);
        %% populate other methods VI NAS CH
        s=struct;
        s.CH=zeros(1,length(peak_train));
        s.NAS=zeros(1,length(peak_train));
        s.VI=zeros(1,length(peak_train));
        [s] = populate_signals(time_stamps_s,s,time_s,burst_CH_s,start_stop_cochlea,visual_inspection,win);
        %% split test data
        test = split_test(s,win,data_raw_resh,other_methods,raw_methods);
        %% plot ideal roc curves
        [h_roc_test,roc_test] = plot_roc_curve(other_methods,test,plot_flag,raw_methods);
        roc_test_all{rec_count,1}=roc_test;
        %% plot histograms with ideal roc thresholds
        h_histograms_test=plot_histograms(test,roc_test,other_methods,raw_methods);
        savefig(h_roc_test,[fig_dir_raw 'roc_test_' num2str(rec_count)])
        savefig(h_histograms_test,[fig_dir_raw 'histogram_test_' num2str(rec_count)])
        close all
        %% percentage of coherence
        perc_NAS_VI(rec_count,1)=100*sum(test.resh.VI & test.resh.NAS)/sum(test.resh.VI);
        perc_CH_VI(rec_count,1)=100*sum(test.resh.VI & test.resh.CH)/sum(test.resh.VI);
        %% figures raw and signal plus histogram
        if plot_flag==1
            plot_raw_histogram=zeros(size(raw_methods));
            for curr_method=1:length(raw_methods)
                method_name=raw_methods{curr_method};
                plot_raw_histogram(curr_method)=plot_raw_hist(data_raw,s_raw,method_name,data_raw_resh,name_exp,th);
            end
        end
        
        %% concatenate n_burst and duration
        raw_dur=zeros(1,length(raw_methods));
        raw_numb=zeros(1,length(raw_methods));
        raw_std=zeros(1,length(raw_methods));
        for curr_method=1:length(raw_methods)
            method_name=raw_methods{curr_method};
            raw_numb(1,curr_method)=size(start_stop_raw.(method_name),1);
            raw_dur(1,curr_method)=mean(start_stop_raw_dur.(method_name));
            raw_std(1,curr_method)=std(start_stop_raw_dur.(method_name));
        end
         %% get burst number and duration
        n_bursts(rec_count,:)=[size(start_stop_cochlea,1) size(visual_inspection,1) raw_numb];
        mean_vals(rec_count,:)=[mean(start_stop_cochlea_dur) mean(start_stop_vi_duration) raw_dur];
        std_vals(rec_count,:)=[std(start_stop_cochlea_dur) std(start_stop_vi_duration) raw_std];
        
        se_vals(rec_count,:)=std_vals(rec_count,:)./sqrt(n_bursts(rec_count,:));
        
        n_bursts_ratio(rec_count,:)=n_bursts(rec_count,:)./n_bursts(rec_count,1);
        mean_vals_ratio(rec_count,:)=mean_vals(rec_count,:)./mean_vals(rec_count,1);
        
        rec_count=rec_count+1;
    end
end

%% plot area under curve roc for all experiments
params={'Sensi','Speci','AROC','Accuracy','PPV','NPV','FNR','FPR','FDR','FOR','F1_score','MCC','BM','MK'};
for curr_param=1:length(params)
    param_name=params{curr_param};
    par.(param_name)=zeros(length(roc_test_all),length(other_methods),length(raw_methods));
end
AROC=zeros(length(roc_test_all),length(other_methods),length(raw_methods));
MCC=zeros(length(roc_test_all),length(other_methods),length(raw_methods));
h=zeros(length(roc_test_all),1);
figure
for curr_recording=1:length(roc_test_all)
    for curr_raw=1:length(raw_methods)
        raw_method_name=raw_methods{curr_raw};
        for curr_method=1:length(other_methods)
            method_name=other_methods{curr_method};
            for curr_param=1:length(params)
                param_name=params{curr_param};
                par.(param_name)(curr_recording,curr_method,curr_raw)=roc_test_all{curr_recording,1}.(method_name).(raw_method_name).param.(param_name);
            end
        end
    end
    h(curr_recording)=subplot(2,7,curr_recording);
    plot(squeeze(par.AROC(curr_recording,:,:)))
    if curr_recording==length(roc_test_all)
        legend(raw_methods,'Interpreter','none')
    end
    xticks(1:length(other_methods))
    xticklabels(other_methods)
end
linkaxes(h,'y')

%% ll recordings at once but separated

h_sub=zeros(42,1);
h_par=zeros(5,1);
k=0;
all_sub=1;
for curr_param=1:length(params)
    if mod(curr_param-1,3)==0
        k=k+1;
        j=1;
        h_par(k)=figure;
    end

    param_name=params{curr_param};
    for curr_other=1:3
        figure(h_par(k));
        h_sub(all_sub,1)=subplot(3,3,curr_other+(3*(j-1)));
        plot([squeeze(par.(param_name)(:,curr_other,:))]','Color',[.9 .9 .9])
        hold on
        boxplot([squeeze(par.(param_name)(:,curr_other,:))],raw_methods)
        ylabel(param_name)
        title(other_methods{curr_other})
        all_sub=all_sub+1;
    end
    j=j+1;
end
h_sub_reshape=reshape(h_sub,3,[]);
for curr_param=1:length(params)
    linkaxes(h_sub_reshape(1:3,curr_param),'y')
end
%% ll recordings at once but separated
figure
h_sub=zeros(6,1);
for curr_other=1:3
    h_sub(curr_other,1)=subplot(2,3,curr_other);
    plot([squeeze(AROC(:,curr_other,:))]','Color',[.9 .9 .9])
    hold on
    boxplot([squeeze(AROC(:,curr_other,:))],raw_methods)
    ylabel('AUC')
    title(other_methods{curr_other})
end

for curr_raw=1:3
    h_sub(curr_raw+3,1)=subplot(2,3,curr_raw+3);
    plot([squeeze(AROC(:,:,curr_raw))]','Color',[.9 .9 .9])
    hold on
    boxplot([squeeze(AROC(:,:,curr_raw))],other_methods)
    ylabel('AUC')
    title(raw_methods{curr_raw},'Interpreter','none')
end
linkaxes(h_sub,'y')


%% plot percentage of VI bins covered by NAS and CH
figure
bar([perc_CH_VI perc_NAS_VI])
legend({'CH','NAS'})
title('% of 50ms VI bursting bins covered')

%% boxplot n_burst
figure
subplot(2,1,1)
plot(n_bursts','Color',[.9 .9 .9])
hold on
boxplot(n_bursts)
xticklabels(all_methods)

subplot(2,1,2)
plot(n_bursts_ratio','Color',[.9 .9 .9])
hold on
boxplot(n_bursts_ratio)
xticklabels(all_methods)

%% boxplot n_burst
figure
subplot(2,1,1)
n_bursts_nan=n_bursts;
n_bursts_nan(n_bursts_nan==1)=NaN;
plot(n_bursts_nan','Color',[.9 .9 .9])
hold on
boxplot(n_bursts_nan)
xticklabels(all_methods)
title('number of burst')
ylabel('number of burst')

subplot(2,1,2)
n_bursts_ratio_nan=n_bursts_nan./n_bursts_nan(:,1);
plot(n_bursts_ratio_nan','Color',[.9 .9 .9])
hold on
boxplot(n_bursts_ratio_nan)
xticklabels(all_methods)
title('number of burst ratio')
ylabel('number of burst ratio')

%% stats
h = kstest(reshape(n_bursts_nan,[],1));
if h==0
    %% n burst parametric
    [p,tbl,stats_n_burst_raw] = anova1(n_bursts_nan,all_methods);
    c_n_burst_raw=multcompare(stats_n_burst_raw,'CType','bonferroni');
else
    %% n burst non parametric
    [p,tbl,stats_n_burst_kw] = kruskalwallis(n_bursts_nan,all_methods);
    c_n_burst_kw=multcompare(stats_n_burst_kw,'CType','tukey-kramer');
end

%%
figure
plot(mean_vals)
%%
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
legend({method_name,'threshold'},'Interpreter','none')
title(['histogram 50ms ' method_name],'Interpreter','none')
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
if ~isempty(start_stop_raw) && size(start_stop_raw,2)==2
    start_stop_raw_dur=start_stop_raw(:,2)-start_stop_raw(:,1);
else
    start_stop_raw=0;
    start_stop_raw_dur=0;
end
end

function [s_raw] = populate_s_raw(raw_methods,peak_train,start_stop_raw,time_s,win)
for curr_method=1:length(raw_methods)
    method_name=raw_methods{curr_method};
    s_raw.(method_name)=zeros(1,length(peak_train));
    if ~(start_stop_raw.(method_name)==0)
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
end

function [start_stop_cochlea_dur,start_stop_cochlea] = get_start_stop_cochlea(cochlea_res,shift)
cochlea_res(cochlea_res.class=='NO SPIKES',:)=[];
cochlea_res.class(cochlea_res.class=='NORMAL')='0';
cochlea_res.class(cochlea_res.class=='ABNORMAL')='1';
cochlea_res.class=str2double(cochlea_res.class);
cochlea_res.Time=cochlea_res.Time-cochlea_res.Time(1); %restart from the first normal value

cochlea_res.Time_stretch=linspace(0,299.9,length(cochlea_res.Time))'; %% I removed 200ms which is roughtly the duration of a window
% cochlea_res.Time_stretch=(0.2:0.2:298.8)'; %% totally wrong at the end

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

function [s] = populate_signals(time_stamps_s,s,time_s,burst_CH_s,start_stop_cochlea,visual_inspection,win)
% plot bursts from r
% plot bursts from matlab (Chiappalone)
for curr_burst=1:size(burst_CH_s,1)
    start_sample_s=time_stamps_s(time_stamps_s==burst_CH_s(curr_burst,1));
    stop_sample_s=time_stamps_s(time_stamps_s==burst_CH_s(curr_burst,2));
    start_sample=find(start_sample_s==time_s);
    stop_sample=find(stop_sample_s==time_s);
    s.CH(start_sample:stop_sample)=1;
end
% plot bursts from cochlea
for curr_burst=1:size(start_stop_cochlea,1)
    start_sample_s=start_stop_cochlea(curr_burst,1);
    stop_sample_s=start_stop_cochlea(curr_burst,2);
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    s.NAS(start_sample:stop_sample)=1;
end
% plot bursts from visual inspection
for curr_burst=1:size(visual_inspection,1)
    start_sample_s=visual_inspection(curr_burst,1);
    stop_sample_s=visual_inspection(curr_burst,2);
    diff_start=abs(start_sample_s-time_s);
    start_sample=find(diff_start==(min(diff_start)));
    diff_stop=abs(stop_sample_s-time_s);
    stop_sample=find(diff_stop==(min(diff_stop)));
    s.VI(start_sample:stop_sample)=1;
end
end

function test = split_test(s,win,data_raw_resh,other_methods,raw_methods)
for curr_raw=1:length(raw_methods)
    raw_method_name=raw_methods{curr_raw};
    for curr_other_method=1:length(other_methods)
        other_method_name=other_methods{curr_other_method};
        %% getting bins with at least a sample "burst"
        test.resh.(other_method_name)=sum(reshape(s.(other_method_name),win,[]))>0;
        %% getting the raw bins with burst non burst
        test.data_burst.(other_method_name).(raw_method_name)=data_raw_resh.(raw_method_name)(test.resh.(other_method_name));
        test.data_no_burst.(other_method_name).(raw_method_name)=data_raw_resh.(raw_method_name)(~test.resh.(other_method_name));
    end
end
end

function [h_roc,roc] = plot_roc_curve(other_methods,test,plot_flag,raw_methods)
%         %% roc on max values per 50ms
counter=1;
h_roc=figure;
for curr_raw=1:length(raw_methods)
    raw_method_name=raw_methods{curr_raw};
    for curr_method=1:length(other_methods)
        subplot(length(raw_methods),length(other_methods),counter)
        method_name=other_methods{curr_method};
        roc.(method_name).(raw_method_name)=roc_curve(test.data_no_burst.(method_name).(raw_method_name)',test.data_burst.(method_name).(raw_method_name)');
        h1=gca;
        auc=h1.Title.String;
        title(['from ' method_name ': ' raw_method_name '; ' auc],'Interpreter','none')
        counter=counter+1;
    end
end

if plot_flag
    savefig(h_roc(curr_method),[fig_dir_raw 'roc_' method_name])
end
end

function h_histograms_test = plot_histograms(test,roc_test,other_methods,raw_methods)
counter=1;
h_histograms_test=figure;
for curr_raw=1:length(raw_methods)
    raw_method_name=raw_methods{curr_raw};
    for curr_method=1:length(other_methods)
        subplot(length(raw_methods),length(other_methods),counter)
        method_name=other_methods{curr_method};
        [n_nb,e_nb]=histcounts(test.data_no_burst.(method_name).(raw_method_name),'Normalization','probability');
        plot(e_nb(1:end-1)-diff(e_nb)/2,n_nb)
        hold on
        [n_b,e_b]=histcounts(test.data_burst.(method_name).(raw_method_name),'Normalization','probability');
        bin_w_old=diff(e_b(1:2));
        bin_w_new=bin_w_old/4;
        [n_b,e_b]=histcounts(test.data_burst.(method_name).(raw_method_name),'Normalization','probability','BinWidth',bin_w_new);
        plot(e_b(1:end-1)+diff(e_b)/2,smooth(n_b,4))
%         plot(e_b(1:end-1)-diff(e_b)/2,n_b)
        th=roc_test.(method_name).(raw_method_name).param.Threshold;
        plot([th th],[0 0.1],'LineWidth',2)
        title(['from ' method_name ': ' raw_method_name],'Interpreter','none')
        xlabel(raw_method_name,'Interpreter','none')
        ylabel('probability')
        legend({'No burst','Burst','Threshold'})
        counter = counter+1;
    end
end

end