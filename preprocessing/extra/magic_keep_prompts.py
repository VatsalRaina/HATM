#! /usr/bin/env python

"""
Converts processed data into equivalent processed data with specified prompt-response pairs kept from the data
"""

import os
import shutil


def main():

    load_path = '/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive/prompts.txt'
    dict_prompt_nums = {}

    f = open(load_path, "r")
    all_samples = f.readlines()
    all_samples_imp = [line.rstrip('\n') for line in all_samples]
    f.close()

    line_num = 0
    for sample in all_samples_imp:
        if sample in dict_prompt_nums:
            dict_prompt_nums[sample][0]+=1
            dict_prompt_nums[sample][1].append(line_num)
        else:
            dict_prompt_nums[sample]=[1,[line_num]]
        line_num+=1

    # Find lines of prompt of interest
    for value in dict_prompt_nums.values():
        #print(value)
        if value[0] == 11048:
            lines_to_keep = value[1]
            break



   
    source_dir = '/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive' 
    destination_dir = '/home/alta/relevance/vr311/data_vatsal/LINSK/Andrey/LINSKuns03evl03_ALL_naive/subsets/ten/shuffled'

    shutil.copyfile(os.path.join(source_dir, 'prompts.txt'), os.path.join(destination_dir, 'prompts.txt'))
    #shutil.copyfile(os.path.join(source_dir, 'prompt_ids.txt'), os.path.join(destination_dir, 'prompt_ids.txt'))
    shutil.copyfile(os.path.join(source_dir, 'responses.txt'), os.path.join(destination_dir, 'responses.txt'))
    shutil.copyfile(os.path.join(source_dir, 'speakers.txt'), os.path.join(destination_dir, 'speakers.txt'))
    shutil.copyfile(os.path.join(source_dir, 'confidences.txt'), os.path.join(destination_dir, 'confidences.txt'))
    shutil.copyfile(os.path.join(source_dir, 'targets.txt'), os.path.join(destination_dir, 'targets.txt'))
    shutil.copyfile(os.path.join(source_dir, 'grades.txt'), os.path.join(destination_dir, 'grades.txt'))

    lines_prompts = open(os.path.join(source_dir, 'prompts.txt'), 'r').readlines()
    #lines_prompt_ids = open(os.path.join(source_dir, 'prompt_ids.txt'), 'r').readlines()
    lines_responses = open(os.path.join(source_dir, 'responses.txt'), 'r').readlines()
    lines_speakers = open(os.path.join(source_dir, 'speakers.txt'), 'r').readlines()
    lines_confidences = open(os.path.join(source_dir, 'confidences.txt'), 'r').readlines()
    lines_targets = open(os.path.join(source_dir, 'targets.txt'), 'r').readlines()
    lines_grades = open(os.path.join(source_dir, 'grades.txt'), 'r').readlines()    
   
    new_prompts = []
    #new_prompt_ids = []
    new_responses = []
    new_speakers = []
    new_confidences = []
    new_targets = []
    new_grades = []

    for curr in lines_to_keep:
        new_prompts.append(lines_prompts[curr])
        #new_prompt_ids.append(lines_prompt_ids[curr])
        new_responses.append(lines_responses[curr])
        new_speakers.append(lines_speakers[curr])
        new_confidences.append(lines_confidences[curr])
        new_targets.append(lines_targets[curr])
        new_grades.append(lines_grades[curr])
    


    out_prompts = open(os.path.join(destination_dir, 'prompts.txt'), 'w')
    out_prompts.writelines(new_prompts)
    out_prompts.close()

    #out_prompt_ids = open(os.path.join(destination_dir, 'prompt_ids.txt'), 'w')
    #out_prompt_ids.writelines(new_prompt_ids)
    #out_prompt_ids.close()
    
    out_responses = open(os.path.join(destination_dir, 'responses.txt'), 'w')
    out_responses.writelines(new_responses)
    out_responses.close()

    out_speakers = open(os.path.join(destination_dir, 'speakers.txt'), 'w')
    out_speakers.writelines(new_speakers)
    out_speakers.close()
    
    out_confidences = open(os.path.join(destination_dir, 'confidences.txt'), 'w')
    out_confidences.writelines(new_confidences)
    out_confidences.close()

    out_targets = open(os.path.join(destination_dir, 'targets.txt'), 'w')
    out_targets.writelines(new_targets)
    out_targets.close()
    
    out_grades = open(os.path.join(destination_dir, 'grades.txt'), 'w')
    out_grades.writelines(new_grades)
    out_grades.close()


if __name__ == '__main__':
    main()
