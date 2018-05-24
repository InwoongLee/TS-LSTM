% row:action class, coulmn:motion_diff
N_P = 10;
N_S = 1;

class_acc_mean = zeros(10,10);

class_acc_mean_window = zeros(10,10);

geomean_range = 1:2;

%% WS_020_TS_20
WS_020 = csvread('./WS_020_TS_20/Auto_20180407_1821.csv',12,1);
test_accuracy_WS_020 = zeros(floor(size(WS_020,1)/11), 1);
for i=1:floor(size(WS_020,1)/11)
    test_accuracy_WS_020(i,1:10) = WS_020(i*11,1:10);
    test_accuracy_WS_020(i,11) = sum(test_accuracy_WS_020(i,1:10))/10;
end

id_WS_020 = find(test_accuracy_WS_020(:,11) == max(test_accuracy_WS_020(:,11)));

id_WS_020_conf = int2str(id_WS_020(1)*100);

path_WS_020 = ['./WS_020_TS_20/view-test-' id_WS_020_conf '.csv'];

confusion_WS_020 = csvread(path_WS_020);
for i=1:size(confusion_WS_020,1)    
    confusion_WS_020(i,11) = confusion_WS_020(i,mod(i-1,10)+1)/sum(confusion_WS_020(i,1:10))*100;
end
 
class_acc_WS_020 = reshape(confusion_WS_020(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,1) = mean(class_acc_WS_020(i,1:10));
end
 
for i=1:10
    class_acc_WS_020(11,i) = mean(class_acc_WS_020(1:10,i));
    class_acc_WS_020(12,i) = geomean(class_acc_WS_020(geomean_range,i));
end

corr_WS_020 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_020(i, :), class_acc_WS_020(j, :));
        corr_WS_020(i,j) = A(1,2);
    end
end


% for i=1:10
%     class_acc_WS_020(i,11) = mean(class_acc_WS_020(i,1:10));
% end


%% WS_040_TS_40
WS_040 = csvread('./WS_040_TS_40/Auto_20180407_2022.csv',12,1);
test_accuracy_WS_040 = zeros(floor(size(WS_040,1)/11), 11);
for i=1:floor(size(WS_040,1)/11)
    test_accuracy_WS_040(i,1:10) = WS_040(i*11,1:10);
    test_accuracy_WS_040(i,11) = sum(test_accuracy_WS_040(i,1:10))/10;
end

id_WS_040 = find(test_accuracy_WS_040(:,11) == max(test_accuracy_WS_040(:,11)));

id_WS_040_conf = int2str(id_WS_040(1)*100);

path_WS_040 = ['./WS_040_TS_40/view-test-' id_WS_040_conf '.csv'];
% 
confusion_WS_040 = csvread(path_WS_040);
for i=1:size(confusion_WS_040,1)    
    confusion_WS_040(i,11) = confusion_WS_040(i,mod(i-1,10)+1)/sum(confusion_WS_040(i,1:10))*100;
end
 
class_acc_WS_040 = reshape(confusion_WS_040(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,2) = mean(class_acc_WS_040(i,1:10));
end
 
for i=1:10
    class_acc_WS_040(11,i) = mean(class_acc_WS_040(1:10,i));
    class_acc_WS_040(12,i) = geomean(class_acc_WS_040(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_040(i,11) = mean(class_acc_WS_040(i,1:10));
% end

corr_WS_040 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_040(i, :), class_acc_WS_040(j, :));
        corr_WS_040(i,j) = A(1,2);
    end
end

%% WS_060_TS_46
WS_060 = csvread('./WS_060_TS_46/Auto_20180409_1206.csv',12,1);
test_accuracy_WS_060 = zeros(floor(size(WS_060,1)/11), 11);
for i=1:floor(size(WS_060,1)/11)
    test_accuracy_WS_060(i,1:10) = WS_060(i*11,1:10);
    test_accuracy_WS_060(i,11) = sum(test_accuracy_WS_060(i,1:10))/10;
end

id_WS_060 = find(test_accuracy_WS_060(:,11) == max(test_accuracy_WS_060(:,11)));

id_WS_060_conf = int2str(id_WS_060(1)*100);

path_WS_060 = ['./WS_060_TS_46/view-test-' id_WS_060_conf '.csv'];
% 
confusion_WS_060 = csvread(path_WS_060);
for i=1:size(confusion_WS_060,1)    
    confusion_WS_060(i,11) = confusion_WS_060(i,mod(i-1,10)+1)/sum(confusion_WS_060(i,1:10))*100;
end
 
class_acc_WS_060 = reshape(confusion_WS_060(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,3) = mean(class_acc_WS_060(i,1:10));
end
 
for i=1:10
    class_acc_WS_060(11,i) = mean(class_acc_WS_060(1:10,i));
    class_acc_WS_060(12,i) = geomean(class_acc_WS_060(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_060(i,11) = mean(class_acc_WS_060(i,1:10));
% end

corr_WS_060 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_060(i, :), class_acc_WS_060(j, :));
        corr_WS_060(i,j) = A(1,2);
    end
end

%% WS_080_TS_60
WS_080 = csvread('./WS_080_TS_60/Auto_20180409_1657.csv',12,1);
test_accuracy_WS_080 = zeros(floor(size(WS_080,1)/11), 11);
for i=1:floor(size(WS_080,1)/11)
    test_accuracy_WS_080(i,1:10) = WS_080(i*11,1:10);
    test_accuracy_WS_080(i,11) = sum(test_accuracy_WS_080(i,1:10))/10;
end

id_WS_080 = find(test_accuracy_WS_080(:,11) == max(test_accuracy_WS_080(:,11)));

id_WS_080_conf = int2str(id_WS_080(1)*100);

path_WS_080 = ['./WS_080_TS_60/view-test-' id_WS_080_conf '.csv'];
% 
confusion_WS_080 = csvread(path_WS_080);
for i=1:size(confusion_WS_080,1)    
    confusion_WS_080(i,11) = confusion_WS_080(i,mod(i-1,10)+1)/sum(confusion_WS_080(i,1:10))*100;
end
 
class_acc_WS_080 = reshape(confusion_WS_080(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,4) = mean(class_acc_WS_080(i,1:10));
end
 
for i=1:10
    class_acc_WS_080(11,i) = mean(class_acc_WS_080(1:10,i));
    class_acc_WS_080(12,i) = geomean(class_acc_WS_080(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_080(i,11) = mean(class_acc_WS_080(i,1:10));
% end
 
corr_WS_080 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_080(i, :), class_acc_WS_080(j, :));
        corr_WS_080(i,j) = A(1,2);
    end
end

%% WS_100_TS_100
WS_100 = csvread('./WS_100_TS_100/Auto_20180408_0859.csv',12,1);
test_accuracy_WS_100 = zeros(floor(size(WS_100,1)/11), 11);
for i=1:floor(size(WS_100,1)/11)
    test_accuracy_WS_100(i,1:10) = WS_100(i*11,1:10);
    test_accuracy_WS_100(i,11) = sum(test_accuracy_WS_100(i,1:10))/10;
end

id_WS_100 = find(test_accuracy_WS_100(:,11) == max(test_accuracy_WS_100(:,11)));

id_WS_100_conf = int2str(id_WS_100(1)*100);

path_WS_100 = ['./WS_100_TS_100/view-test-' id_WS_100_conf '.csv'];
% 
confusion_WS_100 = csvread(path_WS_100);
for i=1:size(confusion_WS_100,1)    
    confusion_WS_100(i,11) = confusion_WS_100(i,mod(i-1,10)+1)/sum(confusion_WS_100(i,1:10))*100;
end
 
class_acc_WS_100 = reshape(confusion_WS_100(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,5) = mean(class_acc_WS_100(i,1:10));
end
 
for i=1:10
    class_acc_WS_100(11,i) = mean(class_acc_WS_100(1:10,i));
    class_acc_WS_100(12,i) = geomean(class_acc_WS_100(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_100(i,11) = mean(class_acc_WS_100(i,1:10));
% end

corr_WS_100 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_100(i, :), class_acc_WS_100(j, :));
        corr_WS_100(i,j) = A(1,2);
    end
end

%% WS_120_TS_80
WS_120 = csvread('./WS_120_TS_80/Auto_20180409_2250.csv',12,1);
test_accuracy_WS_120 = zeros(floor(size(WS_120,1)/11), 11);
for i=1:floor(size(WS_120,1)/11)
    test_accuracy_WS_120(i,1:10) = WS_120(i*11,1:10);
    test_accuracy_WS_120(i,11) = sum(test_accuracy_WS_120(i,1:10))/10;
end

id_WS_120 = find(test_accuracy_WS_120(:,11) == max(test_accuracy_WS_120(:,11)));

id_WS_120_conf = int2str(id_WS_120(1)*100);

path_WS_120 = ['./WS_120_TS_80/view-test-' id_WS_120_conf '.csv'];
% 
confusion_WS_120 = csvread(path_WS_120);
for i=1:size(confusion_WS_120,1)    
    confusion_WS_120(i,11) = confusion_WS_120(i,mod(i-1,10)+1)/sum(confusion_WS_120(i,1:10))*100;
end
 
class_acc_WS_120 = reshape(confusion_WS_120(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,6) = mean(class_acc_WS_120(i,1:10));
end
 
for i=1:10
    class_acc_WS_120(11,i) = mean(class_acc_WS_120(1:10,i));
    class_acc_WS_120(12,i) = geomean(class_acc_WS_120(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_120(i,11) = mean(class_acc_WS_120(i,1:10));
% end

corr_WS_120 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_120(i, :), class_acc_WS_120(j, :));
        corr_WS_120(i,j) = A(1,2);
    end
end

%% WS_140_TS_60
WS_140 = csvread('./WS_140_TS_60/Auto_20180410_0401.csv',12,1);
test_accuracy_WS_140 = zeros(floor(size(WS_140,1)/11), 11);
for i=1:floor(size(WS_140,1)/11)
    test_accuracy_WS_140(i,1:10) = WS_140(i*11,1:10);
    test_accuracy_WS_140(i,11) = sum(test_accuracy_WS_140(i,1:10))/10;
end

id_WS_140 = find(test_accuracy_WS_140(:,11) == max(test_accuracy_WS_140(:,11)));

id_WS_140_conf = int2str(id_WS_140(1)*100);

path_WS_140 = ['./WS_140_TS_60/view-test-' id_WS_140_conf '.csv'];
% 
confusion_WS_140 = csvread(path_WS_140);
for i=1:size(confusion_WS_140,1)    
    confusion_WS_140(i,11) = confusion_WS_140(i,mod(i-1,10)+1)/sum(confusion_WS_140(i,1:10))*100;
end
 
class_acc_WS_140 = reshape(confusion_WS_140(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,7) = mean(class_acc_WS_140(i,1:10));
end
 
for i=1:10
    class_acc_WS_140(11,i) = mean(class_acc_WS_140(1:10,i));
    class_acc_WS_140(12,i) = geomean(class_acc_WS_140(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_140(i,11) = mean(class_acc_WS_140(i,1:10));
% end

corr_WS_140 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_140(i, :), class_acc_WS_140(j, :));
        corr_WS_140(i,j) = A(1,2);
    end
end

%% WS_160_TS_40
WS_160 = csvread('./WS_160_TS_40/Auto_20180410_0943.csv',12,1);
test_accuracy_WS_160 = zeros(floor(size(WS_160,1)/11), 11);
for i=1:floor(size(WS_160,1)/11)
    test_accuracy_WS_160(i,1:10) = WS_160(i*11,1:10);
    test_accuracy_WS_160(i,11) = sum(test_accuracy_WS_160(i,1:10))/10;
end

id_WS_160 = find(test_accuracy_WS_160(:,11) == max(test_accuracy_WS_160(:,11)));

id_WS_160_conf = int2str(id_WS_160(1)*100);

path_WS_160 = ['./WS_160_TS_40/view-test-' id_WS_160_conf '.csv'];
% 
confusion_WS_160 = csvread(path_WS_160);
for i=1:size(confusion_WS_160,1)    
    confusion_WS_160(i,11) = confusion_WS_160(i,mod(i-1,10)+1)/sum(confusion_WS_160(i,1:10))*100;
end
 
class_acc_WS_160 = reshape(confusion_WS_160(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,8) = mean(class_acc_WS_160(i,1:10));
end
 
for i=1:10
    class_acc_WS_160(11,i) = mean(class_acc_WS_160(1:10,i));
    class_acc_WS_160(12,i) = geomean(class_acc_WS_160(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_160(i,11) = mean(class_acc_WS_160(i,1:10));
% end

corr_WS_160 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_160(i, :), class_acc_WS_160(j, :));
        corr_WS_160(i,j) = A(1,2);
    end
end

%% WS_180_TS_20
WS_180 = csvread('./WS_180_TS_20/Auto_20180410_1630.csv',12,1);
test_accuracy_WS_180 = zeros(floor(size(WS_180,1)/11), 11);
for i=1:floor(size(WS_180,1)/11)
    test_accuracy_WS_180(i,1:10) = WS_180(i*11,1:10);
    test_accuracy_WS_180(i,11) = sum(test_accuracy_WS_180(i,1:10))/10;
end

id_WS_180 = find(test_accuracy_WS_180(:,11) == max(test_accuracy_WS_180(:,11)));

id_WS_180_conf = int2str(id_WS_180(1)*100);

path_WS_180 = ['./WS_180_TS_20/view-test-' id_WS_180_conf '.csv'];
% 
confusion_WS_180 = csvread(path_WS_180);
for i=1:size(confusion_WS_180,1)    
    confusion_WS_180(i,11) = confusion_WS_180(i,mod(i-1,10)+1)/sum(confusion_WS_180(i,1:10))*100;
end
 
class_acc_WS_180 = reshape(confusion_WS_180(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,9) = mean(class_acc_WS_180(i,1:10));
end
 
for i=1:10
    class_acc_WS_180(11,i) = mean(class_acc_WS_180(1:10,i));
    class_acc_WS_180(12,i) = geomean(class_acc_WS_180(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_180(i,11) = mean(class_acc_WS_180(i,1:10));
% end

corr_WS_180 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_180(i, :), class_acc_WS_180(j, :));
        corr_WS_180(i,j) = A(1,2);
    end
end

%% WS_200_TS_xx
WS_200 = csvread('./WS_200_TS_xx/Auto_20180411_0038.csv',12,1);
test_accuracy_WS_200 = zeros(floor(size(WS_200,1)/11), 11);
for i=1:floor(size(WS_200,1)/11)
    test_accuracy_WS_200(i,1:10) = WS_200(i*11,1:10);
    test_accuracy_WS_200(i,11) = sum(test_accuracy_WS_200(i,1:10))/10;
end

id_WS_200 = find(test_accuracy_WS_200(:,11) == max(test_accuracy_WS_200(:,11)));

id_WS_200_conf = int2str(id_WS_200(1)*100);

path_WS_200 = ['./WS_200_TS_xx/view-test-' id_WS_200_conf '.csv'];
% 
confusion_WS_200 = csvread(path_WS_200);
for i=1:size(confusion_WS_200,1)    
    confusion_WS_200(i,11) = confusion_WS_200(i,mod(i-1,10)+1)/sum(confusion_WS_200(i,1:10))*100;
end
 
class_acc_WS_200 = reshape(confusion_WS_200(:,11),10,10); % row:action class, coulmn:motion_diff

for i=1:10
    class_acc_mean(i,10) = mean(class_acc_WS_200(i,1:10));
end
 
for i=1:10
    class_acc_WS_200(11,i) = mean(class_acc_WS_200(1:10,i));
    class_acc_WS_200(12,i) = geomean(class_acc_WS_200(geomean_range,i));
end

% for i=1:10
%     class_acc_WS_200(i,11) = mean(class_acc_WS_200(i,1:10));
% end

corr_WS_200 = zeros(10,10);
for i=1:10
    for j=1:10
        A = corrcoef(class_acc_WS_200(i, :), class_acc_WS_200(j, :));
        corr_WS_200(i,j) = A(1,2);
    end
end

corr_WS = (corr_WS_020 + corr_WS_040 + corr_WS_060 + corr_WS_080 + corr_WS_100 + corr_WS_120 + corr_WS_140 + corr_WS_160 + corr_WS_180 + corr_WS_200)/10;
corr_map = (corr_WS<-0.3);


for i=1:10    
    for j=1:10
        corr_map(j,j) = 1;
        geomean_range = find(corr_map(j,:)==1);
        class_acc_WS_020(10+j,i) = geomean(class_acc_WS_020(geomean_range,i));
        class_acc_WS_040(10+j,i) = geomean(class_acc_WS_040(geomean_range,i));
        class_acc_WS_060(10+j,i) = geomean(class_acc_WS_060(geomean_range,i));
        class_acc_WS_080(10+j,i) = geomean(class_acc_WS_080(geomean_range,i));
        class_acc_WS_100(10+j,i) = geomean(class_acc_WS_100(geomean_range,i));
        class_acc_WS_120(10+j,i) = geomean(class_acc_WS_120(geomean_range,i));
        class_acc_WS_140(10+j,i) = geomean(class_acc_WS_140(geomean_range,i));
        class_acc_WS_160(10+j,i) = geomean(class_acc_WS_160(geomean_range,i));
        class_acc_WS_180(10+j,i) = geomean(class_acc_WS_180(geomean_range,i));
        class_acc_WS_200(10+j,i) = geomean(class_acc_WS_200(geomean_range,i));
    end
end

class_acc_metric_020 = zeros(10,10);
class_acc_metric_040 = zeros(10,10);
class_acc_metric_060 = zeros(10,10);
class_acc_metric_080 = zeros(10,10);
class_acc_metric_100 = zeros(10,10);
class_acc_metric_120 = zeros(10,10);
class_acc_metric_140 = zeros(10,10);
class_acc_metric_160 = zeros(10,10);
class_acc_metric_180 = zeros(10,10);
class_acc_metric_200 = zeros(10,10);
for i=1:10
    for j=1:10
        class_acc_metric_020(j, i) = class_acc_WS_020(10+j, i);
        class_acc_metric_040(j, i) = class_acc_WS_040(10+j, i);
        class_acc_metric_060(j, i) = class_acc_WS_060(10+j, i);
        class_acc_metric_080(j, i) = class_acc_WS_080(10+j, i);
        class_acc_metric_100(j, i) = class_acc_WS_100(10+j, i);
        class_acc_metric_120(j, i) = class_acc_WS_120(10+j, i);
        class_acc_metric_140(j, i) = class_acc_WS_140(10+j, i);
        class_acc_metric_160(j, i) = class_acc_WS_160(10+j, i);
        class_acc_metric_180(j, i) = class_acc_WS_180(10+j, i);
        class_acc_metric_200(j, i) = class_acc_WS_200(10+j, i);
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

class_acc_full = [class_acc_metric_020 class_acc_metric_040 class_acc_metric_060 class_acc_metric_080 class_acc_metric_100 class_acc_metric_120 class_acc_metric_140 class_acc_metric_160 class_acc_metric_180 class_acc_metric_200];
Top_num = 20;
action_hyper_W = zeros(10, Top_num);
action_hyper_Mo = zeros(10, Top_num);
action_val = zeros(10, Top_num);
for i = 1:10
%     [val, ind] = sort(class_acc_full(i,:), 'descend');
    [val, ind] = sort(class_acc_full(i,:));
    for j = 1:Top_num
        action_hyper_W(i,j) = ceil(ind(j)/10)*20;
        if mod(ind(j),10) == 0
            action_hyper_Mo(i,j) = 10;
        else
            action_hyper_Mo(i,j) = mod(ind(j),10);
        end
        action_val(i,j) = val(j);
    end            
end

for i=1:10
    class_acc_mean(i,11) = find(class_acc_mean(i,:)==max(class_acc_mean(i,:)));    
end

