#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
    Author : Javier Chiyah-Garcia
    GitHub : https://github.com/JChiyah/what-are-you-referring-to
    Date   : August 2023
    Python : 3.7+

Code released as part of the paper "'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges" accepted at SIGDIAL'23.
"""

import sys
from . import *

import numpy as np


# we assume that the simmc2 data is just outside the current folder (sibling dir)
# sys.path.append('../')
# We use the original evaluation method from the SIMMC2 repository
# from simmc2.model.mm_dst.utils.evaluate_dst import evaluate_from_flat_list
from .evaluate_dst import evaluate_from_flat_list


def _reformat_frame_turn(frame_objects: list):
	"""
	Reformat a turn so it matches the expected format of the original SIMMC2 evaluation script.

	:param frame_objects: list of objects in the turn (pred or true objects).
	:return: format expected by evaluate_from_flat_list
	"""
	frame = {
		'act': [],
		'slots': [],
		'request_slots': [],
		'objects': sorted(frame_objects),
	}
	return [frame]


def _evaluate_from_flat_list_by_model(d_true_flat, d_pred_flat_by_model) -> dict:
	"""
	Evaluate a dataset and get object F1, precision and recall for a dataset.
	It will call the evaluation script for each model given.

	:param d_true_flat: list of true objects
	:param d_pred_flat_by_model: dict of model name -> list of predicted objects
	:return dict: result metrics
	"""
	evaluation = {}
	for model_name, d_pred_flat in d_pred_flat_by_model.items():
		# use the original evaluation method from simmc2
		eval_result = evaluate_from_flat_list(d_true_flat, d_pred_flat)

		# remove everything that doesn't have to do with object f1
		for key in list(eval_result.keys()):        # list avoids error when deleting
			if 'object' not in key:
				del eval_result[key]

		eval_result['entries_evaluated'] = len(d_pred_flat)

		evaluation[model_name] = eval_result

	return evaluation


def evaluate_dataset(dataset: dict, filter_func=None) -> dict:
	"""
	Evaluate a dataset and get object F1, precision and recall for a dataset.
	You can give a filter function to only evaluate a subset of the dataset that
	makes the filter function return True.

	If this function crashes due to zero division, the fix is to change the mm_eva
	

	:param dataset: dataset to evaluate, in the same format as SIMMC2
	:param filter_func: function that takes a turn and returns True if it should be evaluated
	:return dict: result metrics
	"""
	# we need to flatten turns first to evaluate with the same scripts as SIMMC2
	d_true_flattened = []
	d_pred_flattened_by_model_before = {
		m: [] for m in dataset['dialogue_data'][0]['dialogue'][0]['model_outputs'].keys()}
	d_pred_flattened_by_model_after = {
		m: [] for m in d_pred_flattened_by_model_before.keys()}

	evaluating_clarifications = None
	for simmc2_dialogue, simmc2_turn in iterate_over_dataset_entries(dataset):
		if filter_func is not None and not filter_func(simmc2_turn):
			continue # skip as it doesn't pass the filter

		d_true_flattened.append(_reformat_frame_turn(
			simmc2_turn['transcript_annotated']['act_attributes']['objects']))
		if evaluating_clarifications is None:   # first time, set if eval CEs
			evaluating_clarifications = ce.is_ce_turn(simmc2_turn)

		for model_name in d_pred_flattened_by_model_before.keys():

			if evaluating_clarifications:   # calculate before and after CR
				# only doing it once for both before and after CR
				d_pred_flattened_by_model_before[model_name].append(_reformat_frame_turn(
					simmc2_turn['ce'].before_cr_datum['model_outputs'][model_name]['pred_objects']))
				d_pred_flattened_by_model_after[model_name].append(_reformat_frame_turn(
					simmc2_turn['ce'].after_cr_datum['model_outputs'][model_name]['pred_objects']))
			else:
				# evaluating all data in general
				d_pred_flattened_by_model_before[model_name].append(_reformat_frame_turn(
					simmc2_turn['model_outputs'][model_name]['pred_objects']))

	if evaluating_clarifications:   # special return
		if len(d_pred_flattened_by_model_after) != len(d_pred_flattened_by_model_before):
			raise ValueError
		else:
			# return before vs after analysis
			return {
				'Before-CR': _evaluate_from_flat_list_by_model(
					d_true_flattened, d_pred_flattened_by_model_before),
				'After-CR': _evaluate_from_flat_list_by_model(
					d_true_flattened, d_pred_flattened_by_model_after)
			}

	else:
		# return single analysis
		return _evaluate_from_flat_list_by_model(d_true_flattened, d_pred_flattened_by_model_before)


def _extract_target_candidate_objects(
	entry_data: dict, property_key: str, simmc2_metadata: dict, scene_jsons: dict) -> list:
	"""
	Extract the candidate objects for a given turn, based on some property of the
	target object. For instance, if we are talking about a red shirt, we will extract:
	- all shirts in the scene
	- all red objects in the scene
	- etc

	:param entry_data: the turn to extract the candidate objects from
	:param property_key: the property to extract the candidate objects that are similar
	:param simmc2_metadata: the metadata of the SIMMC2 dataset
	:param scene_jsons: the scene jsons of the SIMMC2 dataset
	:return: list of candidate objects
	"""
	# at a given entry, extract the candidate objects based on the target object type
	# e.g., if the target object is a jacket, then all jackets in the scene are candidate objects
	scene_idx_list = [entry_data['scene_idx']]
	if entry_data['previous_scene_idx'] is not None:
		scene_idx_list.append(entry_data['previous_scene_idx'])

	_scene = [scene_jsons[f"{scene_idx}_scene"] for scene_idx in scene_idx_list]
	def all_objects_from_scenes():
		for scene in _scene:
			for _obj in scene['scenes'][0]['objects']:
				yield _obj

	# first case: no target objects
	if len(entry_data['transcript_annotated']['act_attributes']['objects']) == 0:
		return []
	else:
		def _get_object_metadata(object_index):
			for item in all_objects_from_scenes():
				if item['index'] == object_index:
					# now use prefab to find actual item type
					return simmc2_metadata[item['prefab_path']]

			assert f"Could not find object {object_index} in all scene objects {[x['index'] for x in all_objects_from_scenes()]}"

		# 1, 2 or more objects
		target_object_metadata = []
		for item_id in entry_data['transcript_annotated']['act_attributes']['objects']:
			target_object_metadata.append(_get_object_metadata(item_id))

		target_object_types = [x[property_key] for x in target_object_metadata]

		candidates = []
		# if they are the same, then just loop over all items in scene (but target ones) and get their type
		for scene_item in all_objects_from_scenes():
			if scene_item not in entry_data['transcript_annotated']['act_attributes']['objects']:
				if _get_object_metadata(scene_item['index'])[property_key] in target_object_types:
					candidates.append(scene_item)

		return candidates



def extract_candidate_objects(
	dataset: dict, simmc2_metadata: dict, scene_jsons: dict, filter_func=None) -> dict:
	"""
	Extract the candidate objects for a given dataset, based on some property of the
	target object at that turn. For instance, if we are talking about
	a red shirt, we will extract:
	- all shirts in the scene
	- all red objects in the scene
	- etc

	:param dataset: the dataset to extract the candidate objects from
	:param simmc2_metadata: the metadata of the SIMMC2 dataset
	:param scene_jsons: the scene jsons of the SIMMC2 dataset
	:param filter_func: function that takes a turn and returns True if it should
		be evaluated. Use it to extract from different data splits/subsets
	:return: dict of candidate objects, with mean, std and count
	"""
	# define each field that we want to extract, not all objects have all fields
	candidate_objects = {
		'type': [],
		'color': [],
		'brand': [],
		# 'assetType': [],  # only clothes have this field, not furniture
		# 'pattern': [],    # only clothes have this field, not furniture
	}

	for simmc2_dialogue, simmc2_turn in iterate_over_dataset_entries(dataset):
		if filter_func is not None and not filter_func(simmc2_turn):
			continue # skip as it doesn't pass the filter

		for key in candidate_objects.keys():
			candidate_objects[key].append(len(_extract_target_candidate_objects(
				simmc2_turn, key, simmc2_metadata, scene_jsons)))

	# calculate mean & std
	for key in candidate_objects.keys():
		candidate_objects[key] = {
			'mean': np.mean(candidate_objects[key]),
			'std': np.std(candidate_objects[key]),
			'count': len(candidate_objects[key]),
		}

	return candidate_objects
