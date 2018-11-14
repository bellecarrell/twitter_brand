import gzip
import logging
from configs.config import *
from utils.file_io import *
from utils.twitter_user_util import *
import re
import sys
import ijson
import io
import json

logging.basicConfig(level=logging.INFO)

#todo: move cols to global settings.py or config file
timeline_fields = ['id_str', 'created_at', 'text', 'retweet_count','favorite_count']
timeline_cols = ['tweet_id', 'created_at', 'text', 'retweet_count','favorite_count', 'user_id']
info_fields = ['created_at', 'followers_count','friends_count','listed_count', 'statuses_count']
info_cols = ['date','user_id', 'created_at', 'followers_count','friends_count','listed_count', 'statuses_count']


def file_check(filename, mode):
    valid, filetype = False, ''

    if "statuses" in filename and mode in ['timeline','all']:
        valid, filetype = True, 'timeline'
    elif "Info" in filename and mode in ['info','all']:
        valid, filetype = True, 'info'

    return valid, filetype

def get_relevant_fields(json_line,filetype,in_dir):
    if filetype == 'timeline':
        fields = [json_line[field] for field in timeline_fields] + [get_user_id(json_line,filetype)]
    elif filetype == 'info':
        fields = [get_date_from_path(in_dir)] + [get_user_id(json_line,filetype)] + [json_line[field] for field in info_fields]
    return fields

def get_user_id(json,filetype):
    if filetype == 'timeline':
        id = json['user']['id_str']
    elif filetype == 'info':
        id = json['id_str']
    return id

def get_date_from_path(path):
    number_re = re.compile(r'.*?(\d+)')
    return re.findall(number_re,path)[0]

def write_to_user_csv(out_dir,user_id,fields,filetype):
    fname = os.path.join(out_dir,filetype,'{}.csv'.format(user_id))
    if not os.path.isfile(fname):
        if filetype == 'info':
            col_headers = info_cols
        elif filetype == 'timeline':
            col_headers = timeline_cols
        write_row_to_csv(fname,col_headers)
    write_row_to_csv(fname,fields)

def main(in_dir,out_dir,mode):
    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            valid_file_for_mode, filetype = file_check(filename,mode)
            if valid_file_for_mode:
                with gzip.open(os.path.join(dirpath, filename), 'rt') as f:
                    try:
                        for line in f:
                            json_line = json.loads(line)
                            fields = get_relevant_fields(json_line,filetype,dirpath)
                            user_id = get_user_id(json_line,filetype)
                            write_to_user_csv(out_dir,user_id,fields,filetype)
                    except (IOError, EOFError) as e:
                        logging.error(e)


if __name__ == '__main__':
    """
    Write dynamic user information fields and timeline tweets to 
    csv files. A file for every user saved in /info
    and /timeline dirs under the out directory
    """

    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    parser.add_argument("--mode", help="Select which files to consider and write to csv -- user information or timeline. Defaults to both",
                        default='all', choices=['all', 'info', 'timeline'])
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    mode = args.mode

    main(in_dir,out_dir,mode)