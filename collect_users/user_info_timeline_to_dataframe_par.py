import gzip
import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')

import argparse
import re
#import io
import json
import os
import multiprocessing as mp
import numpy as np
import datetime
import time
import random
import pandas as pd

#logging.basicConfig(level=logging.INFO)

#todo: move cols to global settings.py or config file
#timeline_fields = ['id_str', 'created_at', 'text', 'retweet_count','favorite_count']
#timeline_cols = ['tweet_id', 'created_at', 'text', 'retweet_count','favorite_count', 'user_id']
#info_fields = ['created_at', 'followers_count','friends_count','listed_count', 'statuses_count']
#info_cols = ['date','user_id', 'created_at', 'followers_count','friends_count','listed_count', 'statuses_count']

def dump_infos(in_paths, out_dir, num_procs):
    def _write_infos(subset_in_paths, out_path, proc_idx):
        rows = []
        npaths = len(subset_in_paths)
        nrows = 0
        start = time.time()
        
        for pidx, p in enumerate(subset_in_paths):
            if not pidx % 100:
                print('({}s) Info process {}: {}/{}; # rows: {}'.format(int(time.time() - start),
                                                                        proc_idx, pidx,
                                                                        npaths, nrows))
            
            d = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
            ts = int(d)
            try:
                f = gzip.open(p, 'rt', encoding='utf8') if p.endswith('.gz') else open(p, 'rt', encoding='utf8')
                for ln in f:
                    try:
                        info = json.loads(ln)
                        
                        r = [ts, info['id_str'], info['followers_count'],
                             info['friends_count'], info['listed_count'],
                             info['statuses_count'], info['location']]
                        
                        rows.append(r)
                        nrows += 1
                    except Exception as ex:
                        print('problem parsing json: "{}" -- {}'.format(ln, ex))
                f.close()
            except Exception as ex:
                print('problem opening file "{}" -- {}'.format(p, ex))
        
        df = pd.DataFrame(rows, columns=['timestamp', 'user_id', 'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'location'])
        df.to_csv(out_path, sep='\t', encoding='utf8', header=True, index=False, compression='gzip')
    
    path_subsets = [[p for j, p in enumerate(in_paths) if (j%num_procs) == i] for i in range(num_procs)]
    
    out_paths = [os.path.join(out_dir, 'user_info_dynamic.{}.tsv.gz'.format(proc_idx)) for proc_idx in range(num_procs)]
    
    procs = [mp.Process(target=_write_infos, args=(ips, op, i))
             for ips, op, i
             in zip(path_subsets, out_paths, list(range(num_procs)))]
    
    if num_procs > 1:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        _write_infos(path_subsets[0], out_paths[0], 0)
    
    subset_dfs = [pd.read_table(p, sep='\t') for p in out_paths]
    print('read partial info paths')
    joined_df = pd.concat(subset_dfs)
    
    joined_df.to_csv(os.path.join(out_dir, 'user_info_dynamic.tsv.gz'), sep='\t',
                     encoding='utf8', header=True, index=False, compression='gzip')
    print('wrote joined info table')

def dump_tweets(in_paths, out_dir, num_procs):
    def _write_tweets(subset_in_paths, out_path, proc_idx):
        rows = []
        npaths = len(subset_in_paths)
        nrows = 0
        start = time.time()
        
        tweet_idxes = set()
        
        for pidx, p in enumerate(subset_in_paths):
            if not pidx % 10:
                print('({}s) Tweet process {}: {}/{}; # rows: {}'.format(int(time.time() - start),
                                                                        proc_idx, pidx,
                                                                        npaths, nrows))
            
            #d = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
            #ts = int(d)
            try:
                f = gzip.open(p, 'rt', encoding='utf8') if p.endswith('.gz') else open(p, 'rt', encoding='utf8')
                for ln in f:
                    try:
                        tweet = json.loads(ln)
                        
                        # avoid writing duplicates, not perfect since need to deduplicate across processes
                        id = int(tweet['id_str'])
                        if id in tweet_idxes:
                            continue
                        else:
                            tweet_idxes.add(id)

                        mention = 0
                        mention_count = 0
                        url = 0
                        rt = 0
                        reply = 0

                        entities = tweet['entities']
                        if entities['user_mentions']:
                            mention = 1
                            mention_count = len(entities['user_mentions'])
                        if entities['urls']:
                            url = 1
                        if 'retweeted_status' in tweet:
                            rt = 1
                        if type(tweet['in_reply_to_status_id']) is int:
                            reply = 1

                        created_at = int(datetime.datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y').timestamp())
                        r = [tweet['id_str'], created_at, tweet['text'].replace('\r', ' ').replace('\t', ' ').replace('\n', ' '),
                             tweet['user']['id_str'], mention, mention_count, url, rt, reply]
                        
                        rows.append(r)
                        nrows += 1
                    except Exception as ex:
                        print('problem parsing json: "{}" -- {}'.format(ln, ex))
                f.close()
            except Exception as ex:

                print('problem opening file "{}" -- {}'.format(p, ex))
        print('rows writing to df: {}'.format(len(rows)))

        df = pd.DataFrame(rows, columns=['tweet_id', 'created_at', 'text', 'user_id', 'mention', 'mention_count', 'url','rt','reply'])
        #import pdb;pdb.set_trace()
        df.to_csv(out_path, sep='\t', encoding='utf8', header=True, index=False, compression='gzip')
        #import pdb;pdb.set_trace()
    
    path_subsets = [[p for j, p in enumerate(in_paths) if (j%num_procs) == i] for i in range(num_procs)]
    
    out_paths = [os.path.join(out_dir, 'user_tweets.{}.tsv.gz'.format(proc_idx)) for proc_idx in range(num_procs)]
    
    if num_procs > 1:
        procs = [mp.Process(target=_write_tweets, args=(ips, op, i))
                 for ips, op, i
                 in zip(path_subsets, out_paths, list(range(num_procs)))]
        
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        _write_tweets(path_subsets[0], out_paths[0], 0)
    
    #df = pd.read_table('user_tweets.noduplicates.tsv.gz', compression='gzip', encoding='utf8', low_memory=False)
    subset_dfs = [pd.read_table(p, sep='\t', encoding='utf8', low_memory=False, dtype={'tweet_id': str,
                                                                                       'created_at': str,
                                                                                       'text': str,
                                                                                       'user_id': str}, error_bad_lines=False)
                  for p in out_paths]
    print('read partial tweet paths')
    joined_df = pd.concat(subset_dfs)
    
    joined_df_dedup = joined_df.drop_duplicates(subset='tweet_id', keep='first')
    
    # remove lines that were misread
    good_ln_mask = [type(tid)==str and type(cat)==str and type(uid)==str and
                    re.match('\d+', tid) is not None and
                    re.match('\d+', cat) is not None and
                    re.match('\d+', uid) is not None
                    for tid, cat, uid in zip(joined_df_dedup['tweet_id'], joined_df_dedup['created_at'], joined_df_dedup['user_id'])]
    joined_df_dedup = joined_df_dedup[good_ln_mask]
    joined_df_dedup.to_csv(os.path.join(out_dir, 'user_tweets.noduplicates.tsv.gz'), sep='\t',
                           encoding='utf8', header=True, index=False, compression='gzip')
    print('wrote joined tweet table (deduplicated)')

if __name__ == '__main__':
    """
    Write dynamic user information fields and timeline tweets to 
    csv files. A file for every user saved in /info
    and /timeline dirs under the out directory
    """
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--in_dir', required=True, help='input directory')
    parser.add_argument('--out_dir', required=True, help='output directory')
    parser.add_argument('--num_procs', type=int, default=1, help='number of processes to run in parallel')
    parser.add_argument('--max_paths', type=int, default=None, help='for debugging, limit number of input files')
    args = parser.parse_args()
    
    in_dirs = ['/exp/acarrell/twitter_brand_data','/exp/abenton/twitter_brand_data']
    #in_dirs = ['/exp/abenton/twitter_brand_data/1547687722', '/exp/abenton/twitter_brand_data/1540298550', '/exp/abenton/twitter_brand_data/1540313909']
    out_dir = args.out_dir
    num_procs = args.num_procs
    
    # build output directory structure
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    out_info_dir = os.path.join(out_dir, 'info')
    out_timeline_dir = os.path.join(out_dir, 'timeline')
    if not os.path.exists(out_info_dir):
        os.mkdir(out_info_dir)
    if not os.path.exists(out_timeline_dir):
        os.mkdir(out_timeline_dir)
    
    random.seed(12345)
    info_paths = []
    tweet_paths = []

    for in_dir in in_dirs:
        ips = [os.path.join(in_dir, dirpath, p) for dirpath, _, filenames in os.walk(in_dir)
                  for p in filenames if p.startswith('userInfo_')]
        info_paths = info_paths + ips
        tps = [os.path.join(in_dir, dirpath, p) for dirpath, _, filenames in os.walk(in_dir)
                   for p in filenames if p.endswith('.statuses.json.gz')]
        tweet_paths = tweet_paths + tps

    random.shuffle(info_paths)
    random.shuffle(tweet_paths)
    #tweet_paths = tweet_paths[:20]
    print(len(tweet_paths))
    #import pdb;pdb.set_trace()

    print('# info paths: {}; # tweet paths: {}'.format(len(info_paths), len(tweet_paths)))
    if args.max_paths is not None:
        info_paths = info_paths[:args.max_paths]
        tweet_paths = tweet_paths[:args.max_paths]
        print('Restricted to {} paths'.format(args.max_paths))
    
    #dump_infos(info_paths, out_info_dir, num_procs)
    dump_tweets(tweet_paths, out_timeline_dir, num_procs)
