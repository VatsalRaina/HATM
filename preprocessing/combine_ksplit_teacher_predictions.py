from __future__ import print_function, division

import numpy as np
import os
import sys
import argparse
import context

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('--ksplit_teacher_ensembles_dir', type=str, default='/home/alta/BLTSpeaking/top-bkm28/models/ksplit_ensembles_CDE', help='Path to teacher ensembles directory')
parser.add_argument('--save_dir', type=str, default='/home/alta/BLTSpeaking/top-bkm28/data/BLTSgrp24_CDE_ksplit_teacher',
                    help='Path to directory where to save generated epochs')
parser.add_argument('--num_epochs_to_gen', type=int, default=6,
                    help='Number of epochs of training data to generate')


class TeacherPredictions:
    def __init__(self, ensemble_dir_path, eval_name, num_base=10, base_learner_name='atm_seed_', ksplit_num=None):
        self.ensemble_dir_path = ensemble_dir_path
        self.eval_name = eval_name
        self.num_base = num_base
        self.base_learner_name = base_learner_name

        self.ksplit_num = ksplit_num

        self.load_labels_pred_data()

    def load_labels_pred_data(self, predictions_filename='predictions.txt', labels_filename='labels.txt'):  # todo
        ensemble_predictions = []
        all_labels = []
        for seed in range(1, self.num_base + 1):
            model_path = os.path.join(self.ensemble_dir_path, self.base_learner_name + str(seed))
            eval_path = os.path.join(model_path, self.eval_name)
            predictions_path = os.path.join(eval_path, predictions_filename)
            labels_path = os.path.join(eval_path, labels_filename)

            predictions = np.loadtxt(predictions_path, dtype=np.float32)
            labels = np.loadtxt(labels_path, dtype=np.int32)
            ensemble_predictions.append(predictions)
            all_labels.append(labels)
        self.ensemble_predictions = np.stack(ensemble_predictions, axis=1)
        self.labels = all_labels[0]
        for i in range(1, 10):
            assert np.all(self.labels == all_labels[i])
        self.size = len(self.labels)
        return


class Example(object):
    def __init__(self, response, prompt, target, grade, speaker, confidence):
        self.response = response
        self.prompt = prompt
        self.target = target
        self.grade = grade
        self.speaker = speaker
        self.confidence = confidence


class ExamplesSetArrays(object):
    def __init__(self, responses, prompts, targets, grades, speakers, confidences):
        assert type(responses) is np.ndarray
        assert type(prompts) is np.ndarray
        assert type(targets) is np.ndarray
        assert type(grades) is np.ndarray
        assert type(speakers) is np.ndarray
        assert type(confidences) is np.ndarray

        assert len(set([len(responses), len(confidences), len(targets), len(prompts), len(grades), len(speakers)])) == 1

        self.responses = responses
        self.prompts = prompts
        self.targets = targets
        self.grades = grades
        self.speakers = speakers
        self.confidences = confidences

        self.size = len(grades)

    def __getitem__(self, key):
        return ExamplesSetArrays(self.responses[key], self.prompts[key], self.targets[key],
                                 self.grades[key], self.speakers[key],
                                 self.confidences[key])

    def __len__(self):
        return self.size

    def to_list(self):
        responses = list(self.responses)
        prompts = list(self.prompts)
        targets = list(self.targets)
        grades = list(self.grades)
        speakers = list(self.speakers)
        confidences = list(self.confidences)
        return ExamplesSet(responses, prompts, targets, grades, speakers, confidences)


class ExamplesSet(object):
    def __init__(self, responses, prompts, targets, grades, speakers, confidences):
        assert len(set([len(responses), len(confidences), len(targets), len(prompts), len(grades), len(speakers)])) == 1

        self.responses = responses
        self.prompts = prompts
        self.targets = targets
        self.grades = grades
        self.speakers = speakers
        self.confidences = confidences

        self.size = len(grades)

    def __getitem__(self, key):
        if type(key) is int:
            return Example(self.responses[key], self.prompts[key], self.targets[key],
                           self.grades[key], self.speakers[key],
                           self.confidences[key])
        elif type(key) is slice:
            return ExamplesSet(self.responses[key], self.prompts[key], self.targets[key],
                               self.grades[key], self.speakers[key],
                               self.confidences[key])
        else:
            raise AttributeError("Index must be either an integer or a slice object.")

    def __add__(self, other):
        assert type(other) is ExamplesSet
        return ExamplesSet(self.responses + other.responses,
                           self.prompts + other.prompts,
                           self.targets + other.targets,
                           self.grades + other.grades,
                           self.speakers + other.speakers,
                           self.confidences + other.confidences)

    def __len__(self):
        return self.size

    def to_ndarrays(self):
        responses = np.array(self.responses)
        prompts = np.array(self.prompts)
        targets = np.array(self.targets)
        grades = np.array(self.grades)
        speakers = np.array(self.speakers)
        confidences = np.array(self.confidences)
        return ExamplesSetArrays(responses, prompts, targets, grades, speakers, confidences)


def load_from_dataset_txt_files(path):
    # Get the paths to the relevant files
    responses_path = os.path.join(path, 'responses.txt')
    prompts_path = os.path.join(path, 'prompts.txt')
    targets_path = os.path.join(path, 'targets.txt')
    grades_path = os.path.join(path, 'grades.txt')
    speakers_path = os.path.join(path, 'speakers.txt')
    confidences_path = os.path.join(path, 'confidences.txt')

    required_files = [responses_path, prompts_path, targets_path, grades_path, speakers_path, confidences_path]

    # Assert the required files exist
    for path in required_files:
        if not os.path.isfile(path):
            print('File: {} doesn`t exist. Exiting...'.format(path))
            raise Exception('File doesn\' exist')

    # Open All the files
    with open(responses_path, 'r') as d:
        responses = [line.replace('\n', '') for line in d.readlines()]
    with open(confidences_path, 'r') as d:
        confs = [line.replace('\n', '') for line in d.readlines()]
    with open(targets_path, 'r') as t:
        targets = [line.replace('\n', '') for line in t.readlines()]
    with open(prompts_path, 'r') as q:
        prompts = [line.replace('\n', '') for line in q.readlines()]
    with open(grades_path, 'r') as t:
        grades = [line.replace('\n', '') for line in t.readlines()]
    with open(speakers_path, 'r') as s:
        speakers = [line.replace('\n', '') for line in s.readlines()]

    return ExamplesSet(responses, prompts, targets, grades, speakers, confs)

def save_to_txt(path, examples_set):
    responses_path = os.path.join(path, 'responses.txt')
    prompts_path = os.path.join(path, 'prompts.txt')
    targets_path = os.path.join(path, 'targets.txt')
    grades_path = os.path.join(path, 'grades.txt')
    speakers_path = os.path.join(path, 'speakers.txt')
    confidences_path = os.path.join(path, 'confidences.txt')

    responses_str = '\n'.join(examples_set.responses)
    prompts_str = '\n'.join(examples_set.prompts)
    targets_str = '\n'.join(examples_set.targets)
    grades_str = '\n'.join(examples_set.grades)
    speakers_str = '\n'.join(examples_set.speakers)
    confidences_str = '\n'.join(examples_set.confidences)

    # Open and save to all the files
    with open(responses_path, 'w') as f:
        f.write(responses_str)
    with open(confidences_path, 'w') as f:
        f.write(confidences_str)
    with open(targets_path, 'w') as f:
        f.write(targets_str)
    with open(prompts_path, 'w') as f:
        f.write(prompts_str)
    with open(grades_path, 'w') as f:
        f.write(grades_str)
    with open(speakers_path, 'w') as f:
        f.write(speakers_str)

def main(args):

    all_ensembles_dir_path = args.ksplit_teacher_ensembles_dir

    for generated_epoch_num in range(1, args.num_epochs_to_gen + 1):
        print('\t>  Generating epoch number: {}  <\n'.format(generated_epoch_num))

        # Get the teacher predictions for each epoch set
        teacher_predictions = []
        eval_name = 'eval_grp24_epoch' + str(generated_epoch_num)
        for ensemble_num in range(1, 11):
            ensemble_dir_path = os.path.join(all_ensembles_dir_path, 'ensemble_' + str(ensemble_num))

            teacher_predictions.append(TeacherPredictions(ensemble_dir_path, eval_name, ksplit_num=ensemble_num))
            print('Ksplit {} loaded'.format(ensemble_num))

        # Get the labels for each epoch set and assert they are the same as the ones output with the teacher predictions
        # (as a sanity check)
        data_dir = '/home/alta/BLTSpeaking/top-bkm28/data/BLTSgrp24_CDE_10split/dataset_{}/held_out_shuffled' + str(
            generated_epoch_num)
        for ensemble_num in range(10):
            labels_path = os.path.join(data_dir.format(ensemble_num + 1), 'targets.txt')
            epoch_labels = np.loadtxt(labels_path, dtype=np.int32)
            trimmed_size = len(teacher_predictions[ensemble_num].labels)  # Dataset rounded down due to batch_size
            assert np.all(epoch_labels[:trimmed_size] == teacher_predictions[ensemble_num].labels)
            print('Epoch {} labels match'.format(ensemble_num + 1))

        # # Print some predictions for each epoch to make sure they match the labels to the expected amount
        # np.set_printoptions(precision=2, suppress=True)
        #
        # for ensemble_num in range(1, 11):
        #     print('\nTeacher predictions and labels for epoch {}:'.format(ensemble_num))
        #     teacher = teacher_predictions[ensemble_num - 1]
        #     preds_and_labels = np.hstack(
        #         [np.expand_dims(teacher.labels.astype(np.float32), 1), teacher.ensemble_predictions])
        #     print(preds_and_labels[0:4])

        # Get the data files like response, prompts ...
        examples_sets = []
        data_dir = '/home/alta/BLTSpeaking/top-bkm28/data/BLTSgrp24_CDE_10split'
        for ensemble_num in range(10):
            dataset_path = os.path.join(data_dir, 'dataset_' + str(ensemble_num + 1),
                                        'held_out_shuffled' + str(generated_epoch_num))
            ex_set = load_from_dataset_txt_files(dataset_path)

            # Append the trimmed dataset
            trimmed_size = teacher_predictions[ensemble_num].size  # Dataset rounded down due to batch_size
            examples_sets.append(ex_set[:trimmed_size])

            print('Dataset {} retrieved'.format(ensemble_num + 1))

        # Join all the datasets:
        combined_ex_set = reduce((lambda x, y: x + y), examples_sets)

        print("The combined dataset size is: ", len(combined_ex_set))

        # Join the predictions:

        combined_predictions = reduce(lambda x, y: np.vstack([x, y]),
                                      [teacher.ensemble_predictions for teacher in teacher_predictions])
        print("The shape of combined teacher predictions is:", combined_predictions.shape)

        # Shuffle the data
        combined_ex_set_np = combined_ex_set.to_ndarrays()

        shuf_idx = np.random.permutation(np.arange(len(combined_ex_set_np)))
        combined_ex_set_shuffled = combined_ex_set_np[shuf_idx]
        combined_predictions_shuffled = combined_predictions[shuf_idx]
        print('combined predictions after shuffling shape: ', combined_predictions_shuffled.shape)



        # Save everything to a new set:
        save_dir = os.path.join(args.save_dir, 'epoch' + str(generated_epoch_num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        combined_ex_set_shuffled_list = combined_ex_set_shuffled.to_list()
        save_to_txt(save_dir, combined_ex_set_shuffled_list)

        # Save the ensemble predictions as well
        np.savetxt(os.path.join(save_dir, 'teacher_predictions.txt'), combined_predictions_shuffled)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
