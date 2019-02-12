%% get results from r
clear
clc

% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end
cd ..
load('spikes_s.mat')

resultCMA = importfile('result_CMA.csv');
resultHenning = importfile('result_hennig_method.csv');
resultPS = importfile('result_PS_method.csv');
resultRS = importfile('result_RS_method.csv');

fs=1e4;
time_stamps_samples=Spikes_s;
time_stamps_s=time_stamps_samples/fs;

% fileID= fopen('C:\Users\BuccelliLab\Documents\GitHub\burstanalysis\spikes_s.txt','w');
% for curr_sample=1:length(time_stamps_s)
%     fprintf(fileID,'%f\n',time_stamps_s(curr_sample));
% end
% fclose(fileID);

%% plot spikes and burst detection
figure
% plot(time_stamps_samples,ones(size(time_stamps_samples)),'b.')
for curr_spike=1:length(time_stamps_samples)
    curr_sample=time_stamps_samples(curr_spike);
    plot([curr_sample curr_sample],[0 1],'b')
    hold on
end
hold on
for curr_burst=1:height(resultCMA)
    start_sample=time_stamps_samples(resultCMA.beg(curr_burst));
    stop_sample=time_stamps_samples(resultCMA.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2),'ro-')
end
for curr_burst=1:height(resultHenning)
    start_sample=time_stamps_samples(resultHenning.beg(curr_burst));
    stop_sample=time_stamps_samples(resultHenning.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.2,'go-')
end
for curr_burst=1:height(resultPS)
    start_sample=time_stamps_samples(resultPS.beg(curr_burst));
    stop_sample=time_stamps_samples(resultPS.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.4,'mo-')
end
for curr_burst=1:height(resultRS)
    start_sample=time_stamps_samples(resultRS.beg(curr_burst));
    stop_sample=time_stamps_samples(resultRS.end1(curr_burst));
    plot([start_sample stop_sample],zeros(1,2)+.6,'ko-')
end
ylim([-.1 1.1])
title('spike detection comparison')
xlabel('Time [s]')
yticks([0 .2 .4 .6])
yticklabels({'CMA','Henning','PS','RS'})

% xlim([0 .5])