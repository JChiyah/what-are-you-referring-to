#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
    Author : Javier Chiyah-Garcia
    GitHub : https://github.com/JChiyah/what-are-you-referring-to
    Date   : August 2023
    Python : 3.7+

Code released as part of the paper "'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges" accepted at SIGDIAL'23.
"""

import re
import json
from typing import List, Optional

DEBUG = False


# Tags for the clarification exchanges
TAG_COLOUR = 'colour'
TAG_SPATIAL = 'spatial'
TAG_RELATIONAL = 'relational'
TAG_PROPERTY = 'property'
TAG_ITEM = 'type'
TAG_CONFIRMATION = 'confirmation'
TAG_PREVIOUS = 'previous'
TAG_OTHER = 'other'

TAG_INDIVIDUAL_PROPERTY = 'individual_property'
TAG_DIALOGUE_HISTORY = 'dialogue_history'
TAG_RELATIONAL_CONTEXT = 'relational_context'

_TAG_COLLECTION = {
	TAG_INDIVIDUAL_PROPERTY: [TAG_COLOUR, TAG_PROPERTY, TAG_ITEM],
	TAG_DIALOGUE_HISTORY: [TAG_PREVIOUS, TAG_CONFIRMATION],
	TAG_RELATIONAL_CONTEXT: [TAG_SPATIAL, TAG_RELATIONAL],
}
# todo: position vs spatial

TAGS = [TAG_COLOUR, TAG_ITEM, TAG_PROPERTY, TAG_PREVIOUS, TAG_CONFIRMATION, TAG_SPATIAL, TAG_RELATIONAL]

# extracted from metadata
_COLOURS = [
	'gray', 'grey', 'blue', 'red', 'yellow', 'brown', 'black', 'white', 'purple',
	'pink', 'green', 'violet', 'olive', 'maroon', 'orange', 'beige', 'gold',
	'silver', 'teal', 'wooden', 'camo', 'zebra', 'denim'
]
# compound colours, like "light blue" that we check too as part of keywords
_COLOUR_COMPOSITE = [
	'light', 'dark', 'darker', 'lighter'
]

_TAG_KEYWORDS = {
	TAG_COLOUR: _COLOURS + _COLOUR_COMPOSITE,
	TAG_SPATIAL: ['right', 'left', 'bottom', 'top', 'middle', 'center', 'leftmost', 'rightmost'],
    TAG_RELATIONAL: [
		'second', 'cubicle', 'table', 'rack', 'floor', 'lower', 'shelf', 'further', 'above',
		'front', 'behind', 'next', 'wall', 'back', 'cubby', 'closet', 'display', 'row', 'wardrobe',
		'sides', 'cabinet', 'end', 'windows', 'far', 'corner', 'shelves', 'cabinet', 'end', 'on\s(?!that)',
		'closer', 'area', 'farther', 'by the', 'closest'],
	TAG_ITEM: ['jacket', 'blouse', 'shirt', 'jean', 'pants', 'coat', 'sweater', 'tee', 'dress', 'dresses', 'skirt', 'shorts', '[^a-zA-Z]hat([^a-zA-Z]|$)', 'hoodie', 'chair', 'sofa', 'couch', 'rug'],
	TAG_PROPERTY: [
		'sleeve', 'sleeved', 'hanging', 'hangs', 'rating', 'fluffy', 'without the arms', 'taller',        # (?!.*what)rating
		'sleeveless', 'christmas', 'Christmas-looking'],
	TAG_CONFIRMATION: ['okay', 'yeah', 'ok([^a-zA-Z]|$)', 'yes', 'both', 'precisely', 'yep', 'Yep!', 'I do!'],
	TAG_PREVIOUS: [
		'mentioned', 'earlier', 'discussing', 'discussed', 'you put', 'pointed out', 'showed', 'just talking',
		'just added', 'just bought', 'was just', 'just told', 'you recommended', 'was asking', 'added', 'before',
		'shown', 'you just', 'my cart', 'last', 'previously', 'you suggested', 're(?!not) talking', 'just asked', 'you told',
		'you found', 'were just', 'you are recommending', 'first thing i looked at', '.*ed [^\s]* first']
}


def _check_for_keywords_in_utterance(utterance: str, keywords: list):
	for keyword in keywords:
		result = re.search(rf"{keyword}([\s,s.?]|$)", utterance, re.IGNORECASE)
		if result:
			return result
	else:
		return None


def extract_utterance_tags(
	utterance: str, *, gt_referenced_objects=None, fine_grained: bool = False,
	fine_grained_combined: bool = True, is_ambiguous_utterance: bool = False) -> List[str]:
	"""
	Extracts the tags from an utterance. It should be called in as many utterances
	as needed in a CE ie., both user ambiguity and system clarification request,
	as both could contain information.

	:param utterance: the utterance to extract tags from
	:param gt_referenced_objects: the ground truth referenced objects, if available
		this param is mostly used for testing, as we know exactly the objects mentioned
	:param fine_grained: whether to use fine-grained tags or not, default False
	:param is_ambiguous_utterance: whether the utterance is the initial ambiguity utterance
		if it is, we may not return all tags, as we assume some are not possible
		and are just artifacts of the dialogue (e.g., "okay, what is the
		price of the red one?"->confirmation tag will be removed).
	:return: list of tags
	"""
	result_tags = []

	# check number of objects first
	if gt_referenced_objects is not None or '':
		gt_referenced_objects_length = len(json.loads(gt_referenced_objects))
		if gt_referenced_objects_length > 3:
			result_tags.append(f"{gt_referenced_objects_length}-objects")#

	for tag, keywords in _TAG_KEYWORDS.items():
		# check if any keyword in utterance
		result = _check_for_keywords_in_utterance(utterance, keywords)
		if result:
			if tag == TAG_PROPERTY and 'what' in utterance[:result.end()]:
				continue
			elif tag == TAG_PREVIOUS and 'not' in utterance:
				continue # skip
			elif tag in TAG_SPATIAL + TAG_RELATIONAL and 'Could you' in utterance:
				continue # skip
			elif tag == TAG_CONFIRMATION and is_ambiguous_utterance:
				# generally, the initial referential ambiguity utterance does not
				# contain confirmation words, possibly for previous turns instead
				continue
			if DEBUG:
				print(f"Match for {tag} found at {result.start()}-{result.end()}: {result.group()}")
			result_tags.append(tag)

	# cluster together
	combined_tags = []
	for collection in _TAG_COLLECTION:
		if any([tag in result_tags for tag in _TAG_COLLECTION[collection]]):
			combined_tags.append(collection)

	if fine_grained and fine_grained_combined:
		return sort_tags(result_tags + combined_tags)
	elif fine_grained and not fine_grained_combined:
		return sort_tags(result_tags)
	else:
		return sort_tags(combined_tags)


def sort_tags(tags: List[str]) -> List[str]:
	# dummy sorting right now
	return sorted(list(set(tags)))


def count_tags(tags: List[str], *, fine_grained: bool = True) -> int:
	"""
	Counts the number of tags in a list of tags.

	:param tags: list of tags
	:return: dictionary with the count of tags
	"""
	if fine_grained:
		# remove the combined tags
		tags = [tag for tag in tags if tag not in _TAG_COLLECTION.keys()]
	else:
		# remove the fine-grained tags
		tags = [tag for tag in tags if tag in _TAG_COLLECTION.keys()]

	return len(tags)


def test_utterance_tagging():
	"""Tests the utterance tagging with the test set in a file provided."""
	# tests should be inside src/ but change as needed, print(os.getcwd())
	test_path = 'src/tagging_tests.json'
	with open(test_path, 'r') as f_in:
		utterance_tagging_tests = json.load(f_in)

	# invert dictionary (to perform if and only iff for tags)
	test_utterances = {}
	for tag, tests in utterance_tagging_tests.items():
		for i, utterance in enumerate(tests):
			# ignore those that start with #
			if utterance.startswith('#'):
				continue

			if utterance in test_utterances:
				test_utterances[utterance].append(tag)
			else:
				test_utterances[utterance] = [tag] if tag != '' else []

	for utterance, tags in test_utterances.items():
		tags = sort_tags(tags)

		# check utterance and tagging robustness with extra punctuation
		for utt in [utterance, utterance + '.', utterance + '?']:
			tags_extracted = extract_utterance_tags(utt, fine_grained=True, fine_grained_combined=False)
			assert tags == tags_extracted, \
				f"Test failed in utterance '{utt}'!\n test: {tags}\n found: '{tags_extracted}'"

	print(f"{len(test_utterances.keys())} Clarification Exchange Tagging tests passed!")


# test at loading time to ensure tagging works as expected
test_utterance_tagging()
