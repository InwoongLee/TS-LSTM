function [ ] = make_csv_action_5060( root_path, output_path, sklt_info_txt )
+% root_path : raw skeleton txt files from ROSE Lab.
+% output_path : output path for csv
+% sklt_info_txt : skeleton information (In this case, 'Actions_50-60.txt')

fid = fopen(sklt_info_txt, 'r');
missing_sklts = textscan(fid, '%s');
fclose(fid);
 
sklt_list = missing_sklts{1, 1};
num_lines = [];
first_skeleton_start = 1;
first_skeleton_index = 1;
second_skeleton_start = 0;
for skltNo = 1:size(sklt_list, 1)
    if skltNo ~= 1 && sklt_list{skltNo, 1}(1, 1) == 'S'
        second_skeleton_start = skltNo; 
        num_lines = [num_lines; first_skeleton_index, second_skeleton_start - first_skeleton_start - 1];
        first_skeleton_start = second_skeleton_start;
        first_skeleton_index = skltNo;
    elseif skltNo == size(sklt_list, 1)
        num_lines = [num_lines; first_skeleton_index, size(sklt_list, 1) - first_skeleton_start];
    end
end

for skltNo = 1:size(num_lines, 1)
%     disp(skltNo);
    subject_line = num_lines(skltNo, 1) + 1;
    for lineNo = num_lines(skltNo, 1) + 2:num_lines(skltNo, 1) + num_lines(skltNo, 2)
        if sklt_list{lineNo, 1}(1) == 's'
           object_line = lineNo;
           break;
        end
    end
    
    subject_index = subject_line + 1:object_line - 1;
    if skltNo ~= size(num_lines, 1)
        object_index = object_line + 1:num_lines(skltNo + 1, 1) - 1;
    else
        object_index = object_line + 1:size(sklt_list, 1);
    end
    
    sklt_file_name = sklt_list{num_lines(skltNo, 1), 1};
    sklt_info = read_skeleton_file(strcat(root_path, sklt_file_name));
    sklt = zeros(size(sklt_info, 2), 150);
    
    for n = subject_index
        frm_info = sscanf(sklt_list{n, 1}, '%d,%d,%d\r\n');
        if frm_info(3, 1) ~= 0
            for frmNo = frm_info(1, 1):frm_info(2, 1)
                if frm_info(3, 1) <= size(sklt_info(frmNo).bodies, 2)
                    for nodeNo = 1:25
                        sklt(frmNo, 3*(nodeNo-1)+1:3*nodeNo) = [sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).x,...
                            sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).y,...
                            sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).z];
                    end                    
                end
            end
        end
    end
    
    for n = object_index
        frm_info = sscanf(sklt_list{n, 1}, '%d,%d,%d\r\n');
        if frm_info(3, 1) ~= 0
            for frmNo = frm_info(1, 1):frm_info(2, 1)
                if frm_info(3, 1) <= size(sklt_info(frmNo).bodies, 2)                    
                    for nodeNo = 1:25
                        sklt(frmNo, 75 + 3*(nodeNo-1)+1:75 +3*nodeNo) = [sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).x,...
                            sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).y,...
                            sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).z];
                    end
                end
            end
        end
    end
    
    first_num_zero = 0;
    for frmNo = 1:size(sklt, 1)
        if sum(sklt(frmNo, :)) == 0
            first_num_zero = first_num_zero + 1;
        else
            break;
        end
    end
    
    last_num_zero = 0;
    for frmNo = size(sklt, 1):-1:1
        if sum(sklt(frmNo, :)) == 0
            last_num_zero = last_num_zero + 1;
        else
            break;
        end
    end
    
    new_sklt = sklt(first_num_zero + 1:size(sklt, 1) - last_num_zero, :);
    csvwrite(strcat(output_path, sklt_file_name(1:20), '.csv'), new_sklt);    
end

end
