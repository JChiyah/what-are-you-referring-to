#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
    Author : Javier Chiyah-Garcia
    GitHub : https://github.com/JChiyah/what-are-you-referring-to
    Date   : August 2023
    Python : 3.7+

Code released as part of the paper "'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges" accepted at SIGDIAL'23.
"""


from . import tagging


_print_counter = 10


def is_ambiguous_turn(entry_datum) -> bool:
	"""Checks whether a turn is ambiguous as defined in the SIMMC2 dataset."""
	return 'disambiguation_label' in entry_datum and entry_datum['disambiguation_label'] == 1 or ''


class ClarificationExchange:
	"""
	Simple clarification object to store the relevant information of a clarification exchange
	in a way that makes sense to humans to analyse.
	The original SIMMC2 dataset has both user utterance and system response in the same
	turn, but here we find both the turn after the clarification request and
	specify the utterances:
	- referential ambiguity: the initial utterance that contains the ambiguity ("the red one")
	- clarification request: the system utterance that requests clarification ("which one?")
	- clarification response: the user utterance that responds to the clarification request ("the one on the left")
	- resolution: the system utterance that resolves the ambiguity ("that one costs $20")

	This class needs the turn of the ambiguity and the turn after to be created,
	and you have the option to use fine-grained tags (type, position, colour, etc.) or
	the default, which are combined tags (colour+type=Individual Property).
	The default is what is reported in the paper.

	:param before_cr_datum: the turn of the ambiguity
	:param after_cr_datum: the turn after the clarification request
	:param fine_grained_tags: whether to use fine-grained tags or not, default False
	"""

	def __init__(self, before_cr_datum, after_cr_datum, fine_grained_tags=False):
		self.before_cr_datum, self.after_cr_datum = before_cr_datum, after_cr_datum
		self.fine_grained_tags = fine_grained_tags
		# initial user utterance
		self.referential_ambiguity = before_cr_datum['transcript']
		self.c_request = before_cr_datum['system_transcript']
		self.c_response = after_cr_datum['transcript']
		self.resolution = after_cr_datum['system_transcript']
		self.__extract_ce_tags()

		self.pretty_print()

	def pretty_print(self):
		"""Prints the clarification exchange in a human-readable way."""
		global _print_counter
		if _print_counter <= 0:
			return
		print(
			f"  Clarification Exchange\n\t"
			f"USR: {self.referential_ambiguity} | {self.tags_referential_ambiguity}\n\t"
			f"SYS: {self.c_request} | {self.tags_c_request}\n\t"
			f"USR: {self.c_response} | {self.tags_c_response}\n\t"
			f"SYS: {self.resolution}\n\t"
			f"Tags={self.tags}")
		_print_counter -= 1

	def __extract_ce_tags(self):
		"""Extracts the tags from the utterances and saves them in the CE class"""
		# we simply extract the tags from as many utterances as wanted
		self.tags_referential_ambiguity = tagging.extract_utterance_tags(
			self.referential_ambiguity, fine_grained=self.fine_grained_tags)
		self.tags_c_request = tagging.extract_utterance_tags(
			self.c_request, fine_grained=self.fine_grained_tags)
		self.tags_c_response = tagging.extract_utterance_tags(
			self.c_response, fine_grained=self.fine_grained_tags)
		# we don't care about the coreference resolution utterance

		self.tags = tagging.sort_tags(
			self.tags_referential_ambiguity + self.tags_c_request + self.tags_c_response)


def mark_clarification_exchange(ambiguous_turn, response_turn) -> None:
	"""
	Marks a clarification exchange in a turn of the SIMMC2 dataset.
	Note it does not modify the original data, it simply sets a flag pointing
	to a ClarificationExchange class.

	:param ambiguous_turn: the turn of the ambiguity
	:param response_turn: the turn after the clarification request
	:return: None
	"""
	ambiguous_turn['ce_turn'] = 'before'
	response_turn['ce_turn'] = 'after'
	# simple object that makes clearer how CEs are structured in the SIMMC2 dataset
	ambiguous_turn['ce'] = ClarificationExchange(ambiguous_turn, response_turn)


def is_ce_turn(entry_datum: dict) -> bool:
	"""
	Checks whether a turn is part of a clarification exchange.
	To avoid counting CEs twice, we only case about the turn before the CR,
	as the after-CR turn will be automatically analysed at the same time.

	:param entry_datum: the turn to check
	:return bool: whether the tag is in the CE
	"""
	return 'ce_turn' in entry_datum and entry_datum['ce_turn'] == 'before'


def is_tag_in_ce(entry_datum: dict, tag: str) -> bool:
	"""
	Checks whether a tag is in a clarification exchange. It also checks whether
	the turn is part of a clarification exchange.

	:param entry_datum: the turn to check
	:param tag: the tag to check the CE for, see tags in tagging.py
	:return bool: whether the tag is in the CE
	"""
	return is_ce_turn(entry_datum) and tag in entry_datum['ce'].tags
