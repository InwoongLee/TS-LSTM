% row:action class, coulmn:motion_diff
N_P = 8;
N_S = 1;

class_acc_mean = zeros(30,9);

class_acc_mean_window = zeros(30,10);

geomean_range = 1:2;

%% WS_021_TS_21
WS_021 = csvread('./WS_021_TS_021/Auto_20180505_1428.csv',12,1);
test_accuracy_WS_021 = zeros(floor(size(WS_021,1)/11), 1);
for i=1:floor(size(WS_021,1)/11)
    test_accuracy_WS_021(i,1:10) = WS_021(i*11,1:10);
    test_accuracy_WS_021(i,11) = sum(test_accuracy_WS_021(i,1:10))/10;
end

id_WS_021 = find(test_accuracy_WS_021(:,11) == max(test_accuracy_WS_021(:,11)));

id_WS_021_conf = int2str(id_WS_021(1)*100);

path_WS_021 = ['./WS_021_TS_021/view-test-' id_WS_021_conf '.csv'];

confusion_WS_021 = csvread(path_WS_021);
for i=1:size(confusion_WS_021,1)    
    confusion_WS_021(i,11) = confusion_WS_021(i,mod(i-1,30)+1)/sum(confusion_WS_021(i,1:30))*100;
end
 
class_acc_WS_021 = reshape(confusion_WS_021(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,1) = mean(class_acc_WS_021(i,1:10));
end



corr_WS_021 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_021(i, :), class_acc_WS_021(j, :));
        corr_WS_021(i,j) = A(1,2);
    end
end


% for i=1:10
%     class_acc_WS_020(i,11) = mean(class_acc_WS_020(i,1:10));
% end

%% WS_042_TS_42
WS_042 = csvread('./WS_042_TS_042/Auto_20180505_2341.csv',12,1);
test_accuracy_WS_042 = zeros(floor(size(WS_042,1)/11), 1);
for i=1:floor(size(WS_042,1)/11)
    test_accuracy_WS_042(i,1:10) = WS_042(i*11,1:10);
    test_accuracy_WS_042(i,11) = sum(test_accuracy_WS_042(i,1:10))/10;
end

id_WS_042 = find(test_accuracy_WS_042(:,11) == max(test_accuracy_WS_042(:,11)));

id_WS_042_conf = int2str(id_WS_042(1)*100);

path_WS_042 = ['./WS_042_TS_042/view-test-' id_WS_042_conf '.csv'];

confusion_WS_042 = csvread(path_WS_042);
for i=1:size(confusion_WS_042,1)    
    confusion_WS_042(i,11) = confusion_WS_042(i,mod(i-1,30)+1)/sum(confusion_WS_042(i,1:30))*100;
end
 
class_acc_WS_042 = reshape(confusion_WS_042(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,2) = mean(class_acc_WS_042(i,1:10));
end

corr_WS_042 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_042(i, :), class_acc_WS_042(j, :));
        corr_WS_042(i,j) = A(1,2);
    end
end

%% WS_063_TS_52
WS_063 = csvread('./WS_063_TS_052/Auto_20180506_0732.csv',12,1);
test_accuracy_WS_063 = zeros(floor(size(WS_063,1)/11), 1);
for i=1:floor(size(WS_063,1)/11)
    test_accuracy_WS_063(i,1:10) = WS_063(i*11,1:10);
    test_accuracy_WS_063(i,11) = sum(test_accuracy_WS_063(i,1:10))/10;
end

id_WS_063 = find(test_accuracy_WS_063(:,11) == max(test_accuracy_WS_063(:,11)));

id_WS_063_conf = int2str(id_WS_063(1)*100);

path_WS_063 = ['./WS_063_TS_052/view-test-' id_WS_063_conf '.csv'];

confusion_WS_063 = csvread(path_WS_063);
for i=1:size(confusion_WS_063,1)    
    confusion_WS_063(i,11) = confusion_WS_063(i,mod(i-1,30)+1)/sum(confusion_WS_063(i,1:30))*100;
end
 
class_acc_WS_063 = reshape(confusion_WS_063(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,3) = mean(class_acc_WS_063(i,1:10));
end

corr_WS_063 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_063(i, :), class_acc_WS_063(j, :));
        corr_WS_063(i,j) = A(1,2);
    end
end

%% WS_084_TS_84
WS_084 = csvread('./WS_084_TS_084/Auto_20180506_1448.csv',12,1);
test_accuracy_WS_084 = zeros(floor(size(WS_084,1)/11), 1);
for i=1:floor(size(WS_084,1)/11)
    test_accuracy_WS_084(i,1:10) = WS_084(i*11,1:10);
    test_accuracy_WS_084(i,11) = sum(test_accuracy_WS_084(i,1:10))/10;
end

id_WS_084 = find(test_accuracy_WS_084(:,11) == max(test_accuracy_WS_084(:,11)));

id_WS_084_conf = int2str(id_WS_084(1)*100);

path_WS_084 = ['./WS_084_TS_084/view-test-' id_WS_084_conf '.csv'];

confusion_WS_084 = csvread(path_WS_084);
for i=1:size(confusion_WS_084,1)    
    confusion_WS_084(i,11) = confusion_WS_084(i,mod(i-1,30)+1)/sum(confusion_WS_084(i,1:30))*100;
end
 
class_acc_WS_084 = reshape(confusion_WS_084(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,4) = mean(class_acc_WS_084(i,1:10));
end

corr_WS_084 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_084(i, :), class_acc_WS_084(j, :));
        corr_WS_084(i,j) = A(1,2);
    end
end

%% WS_105_TS_62
WS_105 = csvread('./WS_105_TS_062/Auto_20180506_2113.csv',12,1);
test_accuracy_WS_105 = zeros(floor(size(WS_105,1)/11), 1);
for i=1:floor(size(WS_105,1)/11)
    test_accuracy_WS_105(i,1:10) = WS_105(i*11,1:10);
    test_accuracy_WS_105(i,11) = sum(test_accuracy_WS_105(i,1:10))/10;
end

id_WS_105 = find(test_accuracy_WS_105(:,11) == max(test_accuracy_WS_105(:,11)));

id_WS_105_conf = int2str(id_WS_105(1)*100);

path_WS_105 = ['./WS_105_TS_062/view-test-' id_WS_105_conf '.csv'];

confusion_WS_105 = csvread(path_WS_105);
for i=1:size(confusion_WS_105,1)    
    confusion_WS_105(i,11) = confusion_WS_105(i,mod(i-1,30)+1)/sum(confusion_WS_105(i,1:30))*100;
end
 
class_acc_WS_105 = reshape(confusion_WS_105(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,5) = mean(class_acc_WS_105(i,1:10));
end

corr_WS_105 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_105(i, :), class_acc_WS_105(j, :));
        corr_WS_105(i,j) = A(1,2);
    end
end

%% WS_126_TS_41
WS_126 = csvread('./WS_126_TS_041/Auto_20180507_0506.csv',12,1);
test_accuracy_WS_126 = zeros(floor(size(WS_126,1)/11), 1);
for i=1:floor(size(WS_126,1)/11)
    test_accuracy_WS_126(i,1:10) = WS_126(i*11,1:10);
    test_accuracy_WS_126(i,11) = sum(test_accuracy_WS_126(i,1:10))/10;
end

id_WS_126 = find(test_accuracy_WS_126(:,11) == max(test_accuracy_WS_126(:,11)));

id_WS_126_conf = int2str(id_WS_126(1)*100);

path_WS_126 = ['./WS_126_TS_041/view-test-' id_WS_105_conf '.csv'];

confusion_WS_126 = csvread(path_WS_126);
for i=1:size(confusion_WS_126,1)    
    confusion_WS_126(i,11) = confusion_WS_126(i,mod(i-1,30)+1)/sum(confusion_WS_126(i,1:30))*100;
end
 
class_acc_WS_126 = reshape(confusion_WS_126(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,6) = mean(class_acc_WS_126(i,1:10));
end

corr_WS_126 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_126(i, :), class_acc_WS_126(j, :));
        corr_WS_126(i,j) = A(1,2);
    end
end

%% WS_147_TS_20
WS_147 = csvread('./WS_147_TS_020/Auto_20180507_1423.csv',12,1);
test_accuracy_WS_147 = zeros(floor(size(WS_147,1)/11), 1);
for i=1:floor(size(WS_147,1)/11)
    test_accuracy_WS_147(i,1:10) = WS_147(i*11,1:10);
    test_accuracy_WS_147(i,11) = sum(test_accuracy_WS_147(i,1:10))/10;
end

id_WS_147 = find(test_accuracy_WS_147(:,11) == max(test_accuracy_WS_147(:,11)));

id_WS_147_conf = int2str(id_WS_147(1)*100);

path_WS_147 = ['./WS_147_TS_020/view-test-' id_WS_147_conf '.csv'];

confusion_WS_147 = csvread(path_WS_147);
for i=1:size(confusion_WS_147,1)    
    confusion_WS_147(i,11) = confusion_WS_147(i,mod(i-1,30)+1)/sum(confusion_WS_147(i,1:30))*100;
end
 
class_acc_WS_147 = reshape(confusion_WS_147(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,7) = mean(class_acc_WS_147(i,1:10));
end

corr_WS_147 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_147(i, :), class_acc_WS_147(j, :));
        corr_WS_147(i,j) = A(1,2);
    end
end

%% WS_167_TS_100
WS_167 = csvread('./WS_167_TS_100/Auto_20180508_0628.csv',12,1);
test_accuracy_WS_167 = zeros(floor(size(WS_167,1)/11), 1);
for i=1:floor(size(WS_167,1)/11)
    test_accuracy_WS_167(i,1:10) = WS_167(i*11,1:10);
    test_accuracy_WS_167(i,11) = sum(test_accuracy_WS_167(i,1:10))/10;
end

id_WS_167 = find(test_accuracy_WS_167(:,11) == max(test_accuracy_WS_167(:,11)));

id_WS_167_conf = int2str(id_WS_167(1)*100);

path_WS_167 = ['./WS_167_TS_100/view-test-' id_WS_167_conf '.csv'];

confusion_WS_167 = csvread(path_WS_167);
for i=1:size(confusion_WS_167,1)    
    confusion_WS_167(i,11) = confusion_WS_167(i,mod(i-1,30)+1)/sum(confusion_WS_167(i,1:30))*100;
end
 
class_acc_WS_167 = reshape(confusion_WS_167(:,11),30,10); % row:action class, coulmn:motion_diff

for i=1:30
    class_acc_mean(i,8) = mean(class_acc_WS_167(i,1:10));
end

corr_WS_167 = zeros(30,30);
for i=1:30
    for j=1:30
        A = corrcoef(class_acc_WS_167(i, :), class_acc_WS_167(j, :));
        corr_WS_167(i,j) = A(1,2);
    end
end

corr_WS = (corr_WS_021 + corr_WS_042 + corr_WS_063 + corr_WS_084 + corr_WS_105 + corr_WS_126 + corr_WS_147 + corr_WS_167)/8;
corr_map = (corr_WS<-0.3);

class_num = 30;

for i=1:10    
    for j=1:30
        corr_map(j,j) = 1;
        geomean_range = find(corr_map(j,:)==1);
        class_acc_WS_021(class_num+j,i) = geomean(class_acc_WS_021(geomean_range,i));
        class_acc_WS_042(class_num+j,i) = geomean(class_acc_WS_042(geomean_range,i));
        class_acc_WS_063(class_num+j,i) = geomean(class_acc_WS_063(geomean_range,i));
        class_acc_WS_084(class_num+j,i) = geomean(class_acc_WS_084(geomean_range,i));
        class_acc_WS_105(class_num+j,i) = geomean(class_acc_WS_105(geomean_range,i));
        class_acc_WS_126(class_num+j,i) = geomean(class_acc_WS_126(geomean_range,i));
        class_acc_WS_147(class_num+j,i) = geomean(class_acc_WS_147(geomean_range,i));
        class_acc_WS_167(class_num+j,i) = geomean(class_acc_WS_167(geomean_range,i));
    end
end

class_acc_metric_021 = zeros(class_num,10);
class_acc_metric_042 = zeros(class_num,10);
class_acc_metric_063 = zeros(class_num,10);
class_acc_metric_084 = zeros(class_num,10);
class_acc_metric_105 = zeros(class_num,10);
class_acc_metric_126 = zeros(class_num,10);
class_acc_metric_147 = zeros(class_num,10);
class_acc_metric_167 = zeros(class_num,10);
for i=1:10
    for j=1:class_num
        class_acc_metric_021(j, i) = class_acc_WS_021(class_num+j, i);
        class_acc_metric_042(j, i) = class_acc_WS_042(class_num+j, i);
        class_acc_metric_063(j, i) = class_acc_WS_063(class_num+j, i);
        class_acc_metric_084(j, i) = class_acc_WS_084(class_num+j, i);
        class_acc_metric_105(j, i) = class_acc_WS_105(class_num+j, i);
        class_acc_metric_126(j, i) = class_acc_WS_126(class_num+j, i);
        class_acc_metric_147(j, i) = class_acc_WS_147(class_num+j, i);
        class_acc_metric_167(j, i) = class_acc_WS_167(class_num+j, i);
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

class_acc_full = [class_acc_metric_021 class_acc_metric_042 class_acc_metric_063 class_acc_metric_084 class_acc_metric_105 class_acc_metric_126 class_acc_metric_147 class_acc_metric_167];
Top_num = 1;
action_hyper_W = zeros(class_num, Top_num);
action_hyper_Mo = zeros(class_num, Top_num);
action_val = zeros(class_num, Top_num);
for i = 1:class_num
    [val, ind] = sort(class_acc_full(i,:), 'descend');
%     [val, ind] = sort(class_acc_full(i,:));
    for j = 1:Top_num
        action_hyper_W(i,j) = ceil(ind(j)/10)*21;
        if mod(ind(j),10) == 0
            action_hyper_Mo(i,j) = 10;
        else
            action_hyper_Mo(i,j) = mod(ind(j),10);
        end
        action_val(i,j) = val(j);
    end            
end

for i=1:class_num
    a = find(class_acc_mean(i,1:8)==max(class_acc_mean(i,1:8)));    
    class_acc_mean(i,9) = a(1);
end

hyper_para_W = zeros(8, 10);
hyper_para_Mo = zeros(8, 10);
for i=1:class_num
    cnt = 1;
    for j=21:21:168
        if action_hyper_W(i, 1) == j
            hyper_para_W(cnt, action_hyper_Mo(i, 1)) = j;
            hyper_para_Mo(cnt, action_hyper_Mo(i, 1)) = action_hyper_Mo(i, 1);
        end
        cnt = cnt + 1;
    end    
end

[val1, ind1] = sort(action_val(:,1));
exclu = zeros(30,2);
for i=1:30
    exclu(i, 1) = action_hyper_W(ind1(i), 1);
    exclu(i, 2) = action_hyper_Mo(ind1(i), 1);
end


