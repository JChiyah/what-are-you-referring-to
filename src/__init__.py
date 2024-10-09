#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
    Author : Javier Chiyah-Garcia
    GitHub : https://github.com/JChiyah/what-are-you-referring-to
    Date   : August 2023
    Python : 3.7+

Code released as part of the paper "'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges" accepted at SIGDIAL'23.
"""

import copy


def iterate_over_dataset_entries(dataset, limit=None):
	if isinstance(dataset, list):
		for x in dataset:
			yield x

	else:
		for _dialogue_datum in dataset['dialogue_data']:
			for _entry_datum in _dialogue_datum['dialogue']:
				if limit is not None:
					limit -= 1
					if limit < 0:
						return
				yield _dialogue_datum, _entry_datum


def join_dataset_splits(dataset_list: list) -> dict:
	joined_dataset = dataset_list[0]
	for dataset in dataset_list[1:]:
		joined_dataset['dialogue_data'].extend(dataset['dialogue_data'])
	return joined_dataset


def fix_prediction_data_format(dataset):
	dialogue_data = []
	for dialogue_datum, turn in iterate_over_dataset_entries(dataset):

		dialogue_data.append(copy.copy(dialogue_datum))
		if 'pred_objects' not in turn:
			turn['pred_objects'] = turn['transcript_annotated']['act_attributes']['objects']
		dialogue_data[-1]['dialogue'] = [turn]

	dataset['dialogue_data'] = dialogue_data
	return dataset


def get_scene_idx(dialogue_scenes: dict, turn_idx: int) -> tuple:
	# resolve scene id from the dialogue list
	scene_idx_list = []
	for scene_id in reversed(list(dialogue_scenes.keys())):
		if turn_idx >= int(scene_id):
			scene_idx_list.append(dialogue_scenes[scene_id])

	if len(scene_idx_list) == 0 or len(scene_idx_list) > 2:
		raise ValueError

	# current scene, previous scene, image_name
	return scene_idx_list[0], scene_idx_list[1] if len(scene_idx_list) > 1 else None


def format_number(float_number, decimals=2, to_percentage=False):
	if to_percentage:
		float_number *= 100
	formatted_number = str(round(float_number, decimals))
	if formatted_number[0] == '0' and len(formatted_number) > 1 and formatted_number[1] == '.':
		formatted_number = formatted_number[1:]
	# find the decimal point
	for i in range(len(formatted_number)):
		if formatted_number[i] == '.':
			# add zeros to the end
			for j in range(decimals - (len(formatted_number) - i - 1)):
				formatted_number += '0'
			break
	return formatted_number


def format_f1(data):
	return f"{format_number(data['object_f1'], 1, True)} ({format_number(data['object_f1_stderr'])})"


def format_mean(data):
	return f"{format_number(data['mean'], 2)} ({format_number(data['std'])})"


def format_delta(data_before, data_after):
	delta = format_number(data_after['object_f1'] / data_before['object_f1'] - 1, 1, True) if data_before['object_f1'] != 0 else '0'
	if delta[0] != '-':     # is delta negative? add a plus sign otherwise
		delta = f"+{delta}"
	if delta[1] == '.':
		delta = f"{delta[0]}0{delta[1:]}"
	return f"\colourdelta{{{delta}}}"


from . import tagging
from . import ce
