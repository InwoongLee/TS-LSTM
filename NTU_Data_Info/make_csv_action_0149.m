function [ ] = make_csv_action_0149( root_path, output_path, sklt_info_txt )
% root_path : raw skeleton txt files from ROSE Lab.
% output_path : output path for csv
% sklt_info_txt : skeleton information (In this case, 'Actions_01-49.txt')

fid = fopen(sklt_info_txt, 'r');
missing_sklts = textscan(fid, '%s');
fclose(fid);

sklt_list = missing_sklts{1, 1};
for skltNo = 1:size(sklt_list, 1)
    if sklt_list{skltNo, 1}(1, 1) == 'S'
%         disp(sklt_list{skltNo, 1});
        
        if skltNo ~= 1
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
            
            new_sklt = zeros(size(sklt(first_num_zero + 1:size(sklt, 1) - last_num_zero, :), 1), 150);
            new_sklt(:, 1:75) = sklt(first_num_zero + 1:size(sklt, 1) - last_num_zero, :);
            csvwrite(strcat(output_path, sklt_file_name(1:20), '.csv'), new_sklt);
        end
        
        sklt_file_name = sklt_list{skltNo, 1};
        sklt_info = read_skeleton_file(strcat(root_path, sklt_file_name));
        sklt = zeros(size(sklt_info, 2), 75);
    else
%         disp(sklt_list{skltNo, 1});
        frm_info = sscanf(sklt_list{skltNo, 1}, '%d,%d,%d\r\n');
        if frm_info(3, 1) ~= 0
            for frmNo = frm_info(1, 1):frm_info(2, 1)
                for nodeNo = 1:25
                    sklt(frmNo, 3*(nodeNo-1)+1:3*nodeNo) = [sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).x,...
                        sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).y,...
                        sklt_info(frmNo).bodies(frm_info(3, 1)).joints(nodeNo).z];
                end
            end
        end
    end
end
new_sklt = zeros(size(sklt(first_num_zero + 1:size(sklt, 1) - last_num_zero, :), 1), 150);
new_sklt(:, 1:75) = sklt(first_num_zero + 1:size(sklt, 1) - last_num_zero, :);
csvwrite(strcat(output_path, sklt_file_name(1:20), '.csv'), new_sklt);

end

