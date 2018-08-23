import configparser
import os
import argparse
from collections import defaultdict

"""
Contains helper methods for script configuration.
"""


def add_input_output_args(parser):
    parser.add_argument("input", help="input filepath")
    parser.add_argument("output", help="output filepath")
    return parser


def add_descriptive_stats_flag(parser):
    """
    :param parser:
    :return: parser with descriptive stats option for intermediate results
    """
    parser.add_argument("--stats", help="calculate helpful stats such as counts and write to file", action="store_true")
    return parser


def configparser_for_file(filename):
    """
    Returns fleshed out ConfigParser object for given filename.
    :param: filename
    :return: Fleshed ConfigParser for file
    """
    cwd = os.getcwd()
    filepath = '/'.join(cwd.split('/')[:-1]) + '/configs/' + filename
    Config = configparser.ConfigParser()
    Config.read(filepath)
    return Config


'''
Originally from
https://wiki.python.org/moin/ConfigParserExamples
modified to take in config
'''


def config_section_map(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


def occupations_by_sector(config, occupation_section):
    occupations_by_sector = defaultdict(list)
    sectors = config.options(occupation_section)
    for sector in sectors:
        occupations_by_sector[sector] = config.get(occupation_section, sector).split(',')
    return occupations_by_sector
