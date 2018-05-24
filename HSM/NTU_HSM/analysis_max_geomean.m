% row:action class, coulmn:motion_diff
N_P = 10;
N_S = 1;

class_acc_mean = zeros(60,10);

class_acc_mean_window = zeros(60,10);

geomean_range = 1:2;

%% WS_030_TS_30
WS_030 = csvread('./WS_030_TS_30/Auto_20180418_1849.csv',12,1);
test_accuracy_WS_030 = zeros(floor(size(WS_030,1)/11), 1);
for i=1:floor(size(WS_030,1)/11)
    test_accuracy_WS_030(i,1:10) = WS_030(i*11,1:10);
    test_accuracy_WS_030(i,11) = sum(test_accuracy_WS_030(i,1:10))/10;
end

id_WS_030 = find(test_accuracy_WS_030(:,11) == max(test_accuracy_WS_030(:,11)));

id_WS_030_conf = int2str(id_WS_030(1)*100);

path_WS_030 = ['./WS_030_TS_30/view-test-' id_WS_030_conf '.csv'];

confusion_WS_030 = csvread(path_WS_030);
for i=1:size(confusion_WS_030,1)    
    confusion_WS_030(i,11) = confusion_WS_030(i,mod(i-1,60)+1)/sum(confusion_WS_030(i,1:60))*100;
end
 
class_acc_WS_030 = reshape(confusion_WS_030(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_030(i,1:10));
end


corr_WS_030 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_030(i, :), class_acc_WS_030(j, :));
        corr_WS_030(i,j) = A(1,2);
    end
end


% for i=1:10
%     class_acc_WS_020(i,11) = mean(class_acc_WS_020(i,1:10));
% end


%% WS_060_TS_60
WS_060 = csvread('./WS_060_TS_60/Auto_20180418_1914.csv',12,1);
test_accuracy_WS_060 = zeros(floor(size(WS_060,1)/11), 1);
for i=1:floor(size(WS_060,1)/11)
    test_accuracy_WS_060(i,1:10) = WS_060(i*11,1:10);
    test_accuracy_WS_060(i,11) = sum(test_accuracy_WS_060(i,1:10))/10;
end

id_WS_060 = find(test_accuracy_WS_060(:,11) == max(test_accuracy_WS_060(:,11)));

id_WS_060_conf = int2str(id_WS_060(1)*100);

path_WS_060 = ['./WS_060_TS_60/view-test-' id_WS_060_conf '.csv'];

confusion_WS_060 = csvread(path_WS_060);
for i=1:size(confusion_WS_060,1)    
    confusion_WS_060(i,11) = confusion_WS_060(i,mod(i-1,60)+1)/sum(confusion_WS_060(i,1:60))*100;
end
 
class_acc_WS_060 = reshape(confusion_WS_060(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_060(i,1:10));
end


corr_WS_060 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_060(i, :), class_acc_WS_060(j, :));
        corr_WS_060(i,j) = A(1,2);
    end
end

%% WS_090_TS_70
WS_090 = csvread('./WS_090_TS_70/Auto_20180418_1932.csv',12,1);
test_accuracy_WS_090 = zeros(floor(size(WS_090,1)/11), 1);
for i=1:floor(size(WS_090,1)/11)
    test_accuracy_WS_090(i,1:10) = WS_090(i*11,1:10);
    test_accuracy_WS_090(i,11) = sum(test_accuracy_WS_090(i,1:10))/10;
end

id_WS_090 = find(test_accuracy_WS_090(:,11) == max(test_accuracy_WS_090(:,11)));

id_WS_090_conf = int2str(id_WS_090(1)*100);

path_WS_090 = ['./WS_090_TS_70/view-test-' id_WS_060_conf '.csv'];

confusion_WS_090 = csvread(path_WS_090);
for i=1:size(confusion_WS_090,1)    
    confusion_WS_090(i,11) = confusion_WS_090(i,mod(i-1,60)+1)/sum(confusion_WS_090(i,1:60))*100;
end
 
class_acc_WS_090 = reshape(confusion_WS_090(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_090(i,1:10));
end


corr_WS_090 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_090(i, :), class_acc_WS_090(j, :));
        corr_WS_090(i,j) = A(1,2);
    end
end

%% WS_120_TS_60
WS_120 = csvread('./WS_120_TS_60/Auto_20180418_1954.csv',12,1);
test_accuracy_WS_120 = zeros(floor(size(WS_120,1)/11), 1);
for i=1:floor(size(WS_120,1)/11)
    test_accuracy_WS_120(i,1:10) = WS_120(i*11,1:10);
    test_accuracy_WS_120(i,11) = sum(test_accuracy_WS_120(i,1:10))/10;
end

id_WS_120 = find(test_accuracy_WS_120(:,11) == max(test_accuracy_WS_120(:,11)));

id_WS_120_conf = int2str(id_WS_120(1)*100);

path_WS_120 = ['./WS_120_TS_60/view-test-' id_WS_120_conf '.csv'];

confusion_WS_120 = csvread(path_WS_120);
for i=1:size(confusion_WS_120,1)    
    confusion_WS_120(i,11) = confusion_WS_120(i,mod(i-1,60)+1)/sum(confusion_WS_120(i,1:60))*100;
end
 
class_acc_WS_120 = reshape(confusion_WS_120(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_120(i,1:10));
end


corr_WS_120 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_120(i, :), class_acc_WS_120(j, :));
        corr_WS_120(i,j) = A(1,2);
    end
end

%% WS_150_TS_150
WS_150 = csvread('./WS_150_TS_150/Auto_20180420_1628.csv',12,1);
test_accuracy_WS_150 = zeros(floor(size(WS_150,1)/11), 1);
for i=1:floor(size(WS_150,1)/11)
    test_accuracy_WS_150(i,1:10) = WS_150(i*11,1:10);
    test_accuracy_WS_150(i,11) = sum(test_accuracy_WS_150(i,1:10))/10;
end

id_WS_150 = find(test_accuracy_WS_150(:,11) == max(test_accuracy_WS_150(:,11)));

id_WS_150_conf = int2str(id_WS_150(1)*100);

path_WS_150 = ['./WS_150_TS_150/view-test-' id_WS_150_conf '.csv'];

confusion_WS_150 = csvread(path_WS_150);
for i=1:size(confusion_WS_150,1)    
    confusion_WS_150(i,11) = confusion_WS_150(i,mod(i-1,60)+1)/sum(confusion_WS_150(i,1:60))*100;
end
 
class_acc_WS_150 = reshape(confusion_WS_150(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_150(i,1:10));
end


corr_WS_150 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_150(i, :), class_acc_WS_150(j, :));
        corr_WS_150(i,j) = A(1,2);
    end
end

%% WS_180_TS_120
WS_180 = csvread('./WS_180_TS_120/Auto_20180420_1809.csv',12,1);
test_accuracy_WS_180 = zeros(floor(size(WS_180,1)/11), 1);
for i=1:floor(size(WS_180,1)/11)
    test_accuracy_WS_180(i,1:10) = WS_180(i*11,1:10);
    test_accuracy_WS_180(i,11) = sum(test_accuracy_WS_180(i,1:10))/10;
end

id_WS_180 = find(test_accuracy_WS_180(:,11) == max(test_accuracy_WS_180(:,11)));

id_WS_180_conf = int2str(id_WS_180(1)*100);

path_WS_180 = ['./WS_180_TS_120/view-test-' id_WS_180_conf '.csv'];

confusion_WS_180 = csvread(path_WS_180);
for i=1:size(confusion_WS_180,1)    
    confusion_WS_180(i,11) = confusion_WS_180(i,mod(i-1,60)+1)/sum(confusion_WS_180(i,1:60))*100;
end
 
class_acc_WS_180 = reshape(confusion_WS_180(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_180(i,1:10));
end


corr_WS_180 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_180(i, :), class_acc_WS_180(j, :));
        corr_WS_180(i,j) = A(1,2);
    end
end

%% WS_210_TS_90
WS_210 = csvread('./WS_210_TS_90/Auto_20180428_1144.csv',12,1);
test_accuracy_WS_210 = zeros(floor(size(WS_210,1)/11), 1);
for i=1:floor(size(WS_210,1)/11)
    test_accuracy_WS_210(i,1:10) = WS_210(i*11,1:10);
    test_accuracy_WS_210(i,11) = sum(test_accuracy_WS_210(i,1:10))/10;
end

id_WS_210 = find(test_accuracy_WS_210(:,11) == max(test_accuracy_WS_210(:,11)));

id_WS_210_conf = int2str(id_WS_210(1)*100);

path_WS_210 = ['./WS_210_TS_90/view-test-' id_WS_210_conf '.csv'];

confusion_WS_210 = csvread(path_WS_210);
for i=1:size(confusion_WS_210,1)    
    confusion_WS_210(i,11) = confusion_WS_210(i,mod(i-1,60)+1)/sum(confusion_WS_210(i,1:60))*100;
end
 
class_acc_WS_210 = reshape(confusion_WS_210(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_210(i,1:10));
end


corr_WS_210 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_210(i, :), class_acc_WS_210(j, :));
        corr_WS_210(i,j) = A(1,2);
    end
end

%% WS_240_TS_60
WS_240 = csvread('./WS_240_TS_60/Auto_20180428_1142.csv',12,1);
test_accuracy_WS_240 = zeros(floor(size(WS_240,1)/11), 1);
for i=1:floor(size(WS_240,1)/11)
    test_accuracy_WS_240(i,1:10) = WS_240(i*11,1:10);
    test_accuracy_WS_240(i,11) = sum(test_accuracy_WS_240(i,1:10))/10;
end

id_WS_240 = find(test_accuracy_WS_240(:,11) == max(test_accuracy_WS_240(:,11)));

id_WS_240_conf = int2str(id_WS_240(1)*100);

path_WS_240 = ['./WS_240_TS_60/view-test-' id_WS_240_conf '.csv'];

confusion_WS_240 = csvread(path_WS_240);
for i=1:size(confusion_WS_240,1)    
    confusion_WS_240(i,11) = confusion_WS_240(i,mod(i-1,60)+1)/sum(confusion_WS_240(i,1:60))*100;
end
 
class_acc_WS_240 = reshape(confusion_WS_240(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_240(i,1:10));
end


corr_WS_240 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_240(i, :), class_acc_WS_240(j, :));
        corr_WS_240(i,j) = A(1,2);
    end
end

%% WS_270_TS_30
WS_270 = csvread('./WS_270_TS_30/Auto_20180428_1127.csv',12,1);
test_accuracy_WS_270 = zeros(floor(size(WS_270,1)/11), 1);
for i=1:floor(size(WS_270,1)/11)
    test_accuracy_WS_270(i,1:10) = WS_270(i*11,1:10);
    test_accuracy_WS_270(i,11) = sum(test_accuracy_WS_270(i,1:10))/10;
end

id_WS_270 = find(test_accuracy_WS_270(:,11) == max(test_accuracy_WS_270(:,11)));

id_WS_270_conf = int2str(id_WS_270(1)*100);

path_WS_270 = ['./WS_270_TS_30/view-test-' id_WS_270_conf '.csv'];

confusion_WS_270 = csvread(path_WS_270);
for i=1:size(confusion_WS_270,1)    
    confusion_WS_270(i,11) = confusion_WS_270(i,mod(i-1,60)+1)/sum(confusion_WS_270(i,1:60))*100;
end
 
class_acc_WS_270 = reshape(confusion_WS_270(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_270(i,1:10));
end


corr_WS_270 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_270(i, :), class_acc_WS_270(j, :));
        corr_WS_270(i,j) = A(1,2);
    end
end

%% WS_300_TS_100
WS_300 = csvread('./WS_300_TS_100/Auto_20180428_1124.csv',12,1);
test_accuracy_WS_300 = zeros(floor(size(WS_300,1)/11), 1);
for i=1:floor(size(WS_300,1)/11)
    test_accuracy_WS_300(i,1:10) = WS_300(i*11,1:10);
    test_accuracy_WS_300(i,11) = sum(test_accuracy_WS_300(i,1:10))/10;
end

id_WS_300 = find(test_accuracy_WS_300(:,11) == max(test_accuracy_WS_300(:,11)));

id_WS_300_conf = int2str(id_WS_300(1)*100);

path_WS_300 = ['./WS_300_TS_100/view-test-' id_WS_270_conf '.csv'];

confusion_WS_300 = csvread(path_WS_300);
for i=1:size(confusion_WS_300,1)    
    confusion_WS_300(i,11) = confusion_WS_300(i,mod(i-1,60)+1)/sum(confusion_WS_300(i,1:60))*100;
end
 
class_acc_WS_300 = reshape(confusion_WS_300(:,11),60,10); % row:action class, coulmn:motion_diff

for i=1:60
    class_acc_mean(i,1) = mean(class_acc_WS_300(i,1:10));
end


corr_WS_300 = zeros(60,60);
for i=1:60
    for j=1:60
        A = corrcoef(class_acc_WS_300(i, :), class_acc_WS_300(j, :));
        corr_WS_300(i,j) = A(1,2);
    end
end



corr_WS = (corr_WS_030 + corr_WS_060 + corr_WS_090 + corr_WS_120 + corr_WS_150 + corr_WS_180 + corr_WS_210 + corr_WS_240 + corr_WS_270 + corr_WS_300)/10;
corr_map = (corr_WS<-0.3);


for i=1:10    
    for j=1:60
        corr_map(j,j) = 1;
        geomean_range = find(corr_map(j,:)==1);
        class_acc_WS_030(60+j,i) = geomean(class_acc_WS_030(geomean_range,i));
        class_acc_WS_060(60+j,i) = geomean(class_acc_WS_060(geomean_range,i));
        class_acc_WS_090(60+j,i) = geomean(class_acc_WS_090(geomean_range,i));
        class_acc_WS_120(60+j,i) = geomean(class_acc_WS_120(geomean_range,i));
        class_acc_WS_150(60+j,i) = geomean(class_acc_WS_150(geomean_range,i));
        class_acc_WS_180(60+j,i) = geomean(class_acc_WS_180(geomean_range,i));
        class_acc_WS_210(60+j,i) = geomean(class_acc_WS_210(geomean_range,i));
        class_acc_WS_240(60+j,i) = geomean(class_acc_WS_240(geomean_range,i));
        class_acc_WS_270(60+j,i) = geomean(class_acc_WS_270(geomean_range,i));
        class_acc_WS_300(60+j,i) = geomean(class_acc_WS_300(geomean_range,i));
    end
end

class_num = 60;

class_acc_metric_030 = zeros(class_num,10);
class_acc_metric_060 = zeros(class_num,10);
class_acc_metric_090 = zeros(class_num,10);
class_acc_metric_120 = zeros(class_num,10);
class_acc_metric_150 = zeros(class_num,10);
class_acc_metric_180 = zeros(class_num,10);
class_acc_metric_210 = zeros(class_num,10);
class_acc_metric_240 = zeros(class_num,10);
class_acc_metric_270 = zeros(class_num,10);
class_acc_metric_300 = zeros(class_num,10);
for i=1:10
    for j=1:class_num
        class_acc_metric_030(j, i) = class_acc_WS_030(class_num+j, i);
        class_acc_metric_060(j, i) = class_acc_WS_060(class_num+j, i);
        class_acc_metric_090(j, i) = class_acc_WS_090(class_num+j, i);
        class_acc_metric_120(j, i) = class_acc_WS_120(class_num+j, i);
        class_acc_metric_150(j, i) = class_acc_WS_150(class_num+j, i);
        class_acc_metric_180(j, i) = class_acc_WS_180(class_num+j, i);
        class_acc_metric_210(j, i) = class_acc_WS_210(class_num+j, i);
        class_acc_metric_240(j, i) = class_acc_WS_240(class_num+j, i);
        class_acc_metric_270(j, i) = class_acc_WS_270(class_num+j, i);
        class_acc_metric_300(j, i) = class_acc_WS_300(class_num+j, i);
%         class_acc_metric_020(j, i) = geomean([class_acc_WS_020(j, i), class_acc_WS_020(10+j, i)]);
%         class_acc_metric_040(j, i) = geomean([class_acc_WS_040(j, i), class_acc_WS_040(10+j, i)]);
%         class_acc_metric_060(j, i) = geomean([class_acc_WS_060(j, i), class_acc_WS_060(10+j, i)]);
%         class_acc_metric_080(j, i) = geomean([class_acc_WS_080(j, i), class_acc_WS_080(10+j, i)]);
%         class_acc_metric_100(j, i) = geomean([class_acc_WS_100(j, i), class_acc_WS_100(10+j, i)]);
%         class_acc_metric_120(j, i) = geomean([class_acc_WS_120(j, i), class_acc_WS_120(10+j, i)]);
%         class_acc_metric_140(j, i) = geomean([class_acc_WS_140(j, i), class_acc_WS_140(10+j, i)]);
%         class_acc_metric_160(j, i) = geomean([class_acc_WS_160(j, i), class_acc_WS_160(10+j, i)]);
%         class_acc_metric_180(j, i) = geomean([class_acc_WS_180(j, i), class_acc_WS_180(10+j, i)]);
%         class_acc_metric_200(j, i) = geomean([class_acc_WS_200(j, i), class_acc_WS_200(10+j, i)]);
%         class_acc_metric_020(j, i) = (class_acc_WS_020(j, i) + class_acc_WS_020(10+j, i))/2;
%         class_acc_metric_040(j, i) = (class_acc_WS_040(j, i) + class_acc_WS_040(10+j, i))/2;
%         class_acc_metric_060(j, i) = (class_acc_WS_060(j, i) + class_acc_WS_060(10+j, i))/2;
%         class_acc_metric_080(j, i) = (class_acc_WS_080(j, i) + class_acc_WS_080(10+j, i))/2;
%         class_acc_metric_100(j, i) = (class_acc_WS_100(j, i) + class_acc_WS_100(10+j, i))/2;
%         class_acc_metric_120(j, i) = (class_acc_WS_120(j, i) + class_acc_WS_120(10+j, i))/2;
%         class_acc_metric_140(j, i) = (class_acc_WS_140(j, i) + class_acc_WS_140(10+j, i))/2;
%         class_acc_metric_160(j, i) = (class_acc_WS_160(j, i) + class_acc_WS_160(10+j, i))/2;
%         class_acc_metric_180(j, i) = (class_acc_WS_180(j, i) + class_acc_WS_180(10+j, i))/2;
%         class_acc_metric_200(j, i) = (class_acc_WS_200(j, i) + class_acc_WS_200(10+j, i))/2;
    end
end

class_acc_full = [class_acc_metric_030 class_acc_metric_060 class_acc_metric_090 class_acc_metric_120 class_acc_metric_150 class_acc_metric_180 class_acc_metric_210 class_acc_metric_240 class_acc_metric_270 class_acc_metric_300];
Top_num = 10;
action_hyper_W = zeros(class_num, Top_num);
action_hyper_Mo = zeros(class_num, Top_num);
action_val = zeros(class_num, Top_num);
for i = 1:class_num
    [val, ind] = sort(class_acc_full(i,:), 'descend');
%     [val, ind] = sort(class_acc_full(i,:));
    for j = 1:Top_num
        action_hyper_W(i,j) = ceil(ind(j)/10)*30;
        if mod(ind(j),10) == 0
            action_hyper_Mo(i,j) = 10;
        else
            action_hyper_Mo(i,j) = mod(ind(j),10);
        end
        action_val(i,j) = val(j);
    end            
end

for i=1:class_num
    class_acc_mean(i,11) = find(class_acc_mean(i,:)==max(class_acc_mean(i,:)));    
end

hyper_para_W = zeros(10, 10);
hyper_para_Mo = zeros(10, 10);
for i=1:class_num
    cnt = 1;
    for j=30:30:300
        if action_hyper_W(i, 1) == j
            hyper_para_W(cnt, action_hyper_Mo(i, 1)) = j;
            hyper_para_Mo(cnt, action_hyper_Mo(i, 1)) = action_hyper_Mo(i, 1);
        end
        cnt = cnt + 1;
    end    
end

[val1, ind1] = sort(action_val(:,1));
exclu = zeros(60,2);
for i=1:60
    exclu(i, 1) = action_hyper_W(ind1(i), 1);
    exclu(i, 2) = action_hyper_Mo(ind1(i), 1);
end

cluster_action = zeros(60,60);
for i=1:60    
    cluster_action(i,1:size(find(corr_map(i,:)==1),2)) = find(corr_map(i,:)==1);
end