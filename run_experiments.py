#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
    Author : Javier Chiyah-Garcia
    GitHub : https://github.com/JChiyah/what-are-you-referring-to
    Date   : August 2023
    Python : 3.7+

Code released as part of the paper "'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges" accepted at SIGDIAL'23.

Main script to run experiments.
"""

import os
import sys
import json
import glob

from tqdm import tqdm

# we assume that the simmc2 data is just outside the current folder (sibling dir)
sys.path.append('../')
# imported here to make sure it works, but used in src.evaluation.py
from simmc2.model.mm_dst.utils.evaluate_dst import evaluate_from_flat_list
SIMMC2_FOLDER = '../simmc2/data'

from src import *
from src import evaluation

DATA_FOLDER = 'data'
#%%

# read original SIMMC 2.0 data
simmc2_metadata = {}
for domain in tqdm(['fashion', 'furniture'], desc='Reading Metadata'):
	with open(os.path.join(SIMMC2_FOLDER, f"{domain}_prefab_metadata_all.json"), 'r') as f_in:
		simmc2_metadata = {**simmc2_metadata, **json.load(f_in)}

simmc2_scenes_jsons ={}
_files = glob.glob(f"{SIMMC2_FOLDER}/simmc2_scene_jsons_dstc10_public/*.json")
for file in tqdm(_files, desc='  JSON scenes'):
	with open(file, "r") as f_in:
		simmc2_scenes_jsons[os.path.splitext(os.path.basename(file))[0]] = json.load(f_in)

with open(os.path.join(SIMMC2_FOLDER, 'simmc2_dials_dstc10_devtest.json'), 'r') as f_in:
	simmc2_dataset = json.load(f_in)
#%%

# read model output files
model_outputs = {}

for subdir, dirs, files in os.walk(DATA_FOLDER):
	if 'coref-pred-devtest-mini.json' in files:
		with open(f"{subdir}/coref-pred-devtest-mini.json", 'r') as f_in:
			model_name = subdir.split('/')[-1]
			model_outputs[model_name] = json.load(f_in)
			# make sure that the dialogues have at most 1 turn (it's a specific format from SIMMC2 challenge)
			for dialogue in model_outputs[model_name]['dialogue_data'][:3]:
				if len(dialogue['dialogue']) > 1:
					# we need to fix this dataset!
					model_outputs[model_name] = fix_prediction_data_format(model_outputs[model_name])
					break

# sort dictionary by key
model_outputs = dict(sorted(model_outputs.items(), key=lambda item: item[0]))

# This is another variant of the Baseline GPT-2 from the challenge without the MultiModal help
del model_outputs['Baseline_GPT2_noMM']
# We process the output of this model, but Team9 skipped predictions for ambiguous (Before-CR) turns and
# that needs to be changed in their original code. We left here for comparing All and After-CR turns
del model_outputs['Team9']

all_models = list(model_outputs.keys())
print(f"Loaded outputs: {all_models}")

# do some pre-processing on the original simmc2 data
last_ambiguous_turn = None

# we can iterate the original data easily, but the model output data is formatted
# in a slightly different way, so we need the t_index to access the turn
print('Preprocessing dataset and printing example Clarification Exchanges (CEs)')
for t_index, simmc2_datum in enumerate(iterate_over_dataset_entries(simmc2_dataset)):
	simmc2_dialogue, simmc2_turn = simmc2_datum
	simmc2_turn['model_outputs'] = {}
	simmc2_turn['scene_idx'], simmc2_turn['previous_scene_idx'] = get_scene_idx(
		simmc2_dialogue['scene_ids'], simmc2_turn['turn_idx'])

	for model_name, model_output in model_outputs.items():
		pred_dialogue = model_output['dialogue_data'][t_index]
		pred_turn = pred_dialogue['dialogue'][0]	# in model outputs, there is at most 1 turn per dialogue

		# check that we are in the same dialogue and turn in all the model outputs
		assert simmc2_dialogue['dialogue_idx'] == pred_dialogue['dialogue_idx'] \
		       and simmc2_turn['turn_idx'] == pred_turn['turn_idx'], \
			f"Model output {model_name} does not match the original SIMMC2 data: " \
			f"Dialogue {simmc2_dialogue['dialogue_idx']} Turn {simmc2_turn['turn_idx']} " \
			f"Dialogue {pred_dialogue['dialogue_idx']} Turn {pred_turn['turn_idx']}"

		# add the model output to the turn as 'synced' or joint data
		simmc2_turn['model_outputs'][model_name] = pred_turn

	# check for Clarification Exchanges
	if ce.is_ambiguous_turn(simmc2_turn):
		last_ambiguous_turn = simmc2_dialogue, simmc2_turn
	elif last_ambiguous_turn is not None:
		# this is the turn after the CR, make sure they are the same dialogue
		if last_ambiguous_turn[0]['dialogue_idx'] == simmc2_dialogue['dialogue_idx']:
			ce.mark_clarification_exchange(
				ambiguous_turn=last_ambiguous_turn[1], response_turn=simmc2_turn)
		last_ambiguous_turn = None

# last_ambiguous_turn = None	# reset for new dialogues, probs redundant
#%%
# Define dataset splits as filters through the original SIMMC2 data
all_splits = [
	('All Turns', None),
	# ('Unambiguous Turns (All - CR Turns)', lambda x: not ce.is_ce_turn(x)),
	('CR Turns', lambda x: ce.is_ce_turn(x)),
	('Individual Property', lambda x: ce.is_tag_in_ce(x, tagging.TAG_INDIVIDUAL_PROPERTY)),
	('Dialogue History', lambda x: ce.is_tag_in_ce(x, tagging.TAG_DIALOGUE_HISTORY)),
	('Relational Context', lambda x: ce.is_tag_in_ce(x, tagging.TAG_RELATIONAL_CONTEXT)),
]
#%%
# Create the Evaluation Table 2 from the paper by analysing the data and printing to a LaTex format
print(f"Evaluation Results Table (Latex)\n{'=' * 24}\n")

def format_row_as_latex(_analysis):
	final_str = []
	for _model_name in all_models:
		if 'Before-CR' in _analysis:
			final_str += [
				f"{format_f1(_analysis['Before-CR'][_model_name]):<14}",
				f"{format_f1(_analysis['After-CR'][_model_name]):<14}",
				f"{format_delta(_analysis['Before-CR'][_model_name], _analysis['After-CR'][_model_name]):<10}"]
		else:
			final_str += [f"\multicolumn{{2}}{{c}}{{{format_f1(_analysis[_model_name])}}} ", ' '*10]

	return ' & '.join(final_str) + ' \\\\'

analysis = {}
for split_name, filter_func in all_splits:
	if split_name == 'All Turns':
		# first time! print headers
		headers = ["\multicolumn{3}{c}{" + x + "}}" for x in all_models]
		subheaders = ['Before-CR     ', 'After-CR      ', '$\\Delta$  '] * len(all_models)
		print(' & '.join(['Model' + ' '*15] + [f"{h:<44}" for h in headers]) + ' \\\\')
		print(f"{' & '.join(['Split' + ' '*15] + subheaders)} \\\\")

	# use the filter func to create a split of the data, then check the results of each model
	analysis[split_name] = evaluation.evaluate_dataset(simmc2_dataset, filter_func)
	print(f"{split_name:<20} & {format_row_as_latex(analysis[split_name])}")
#%%
# Create the Candidate Objects Table from Appendix A.2
print(f"Candidate Objects Table (Latex)\n{'=' * 23}\n")

headers = ['Split' + ' '*15, 'Mean Candidate Objects Type (SD)  ', 'Mean Candidate Objects Colour (SD)', 'Entries']

for split_name, filter_func in all_splits:
	if split_name == 'All Turns':
		# first time! print headers
		print(f"{' & '.join(headers)} \\\\")

	analysis = evaluation.extract_candidate_objects(simmc2_dataset, simmc2_metadata, simmc2_scenes_jsons, filter_func)

	print(f"{split_name:<20} & {format_mean(analysis['type'])}{' '*23} & {format_mean(analysis['color'])}{' '*23} & {analysis['type']['count']} \\\\")
# & {format_mean(analysis['brand'])}{' '*22} - used rarely in clarifications, so skipped from table
