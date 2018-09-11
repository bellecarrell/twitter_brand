
'''
Pulls data for brand-like, self-promoting users.

Adrian Benton
8/25/2018
'''
#!/usr/bin/env python

import argparse
import os, random, re, time
from multiprocessing import Process

#TOP_DIR = '/exp/abenton/twitter_brand_data/'
TOP_DIR = '/exp/acarrell/twitter_brand/user_infos/'
#if not os.path.exists(TOP_DIR): os.mkdir(TOP_DIR)

KEY = 'dl1'
KEY_PATH = '/exp/abenton/twitter_collection/keys/{}_keys.txt'.format(KEY)
ACCESS_TOKEN_PATH = '/exp/abenton/twitter_collection/access_tokens/access_token_{}.txt'.format(KEY)

BRAND_USER_DIR = '/exp/acarrell/twitter_brand/blogger_2018/17_allids/'

brand_user_re = re.compile('(?P<job>\w+)\.users\.txt')

WORKING_KEYS = ['ac', 'dl1', 'dl2', 'dl3', 'dl4', 'home', 't1', 't2', 't3', 't4',
                't5', 't6', 't7', 't8', 'tv2', 'tv3', 'tv4', 'tv5', 'tv6']
WORKING_KEY_PATHS = ['/exp/abenton/twitter_collection/keys/{}_keys.txt'.format(key)
                     for key in WORKING_KEYS
]
WORKING_ACCESS_TOKEN_PATHS = [
  '/exp/abenton/twitter_collection/access_tokens/access_token_{}.txt'.format(key)
  for key in WORKING_KEYS
]
N_KEYS = len(WORKING_KEYS)

def getTimeString():
  t = time.localtime()
  return '%d-%d-%d-%d-%d-%d' % (t.tm_year, t.tm_mon,
                                t.tm_mday, t.tm_hour,
                                t.tm_min, t.tm_sec)

def make_dirs(job):
  time_stamp = int(time.time())
  time_dir = os.path.join(TOP_DIR, '{}'.format(time_stamp))
  if not os.path.exists(time_dir):
    os.mkdir(time_dir)
  
  job_dir = os.path.join(time_dir, job)
  if not os.path.exists(job_dir):
    os.mkdir(job_dir)
  
  info_dir = os.path.join(job_dir, 'user_information')
  tweet_dir = os.path.join(job_dir, 'user_timeline')
  #network_dir = os.path.join(time_dir, 'networks')
  
  for d in [info_dir, tweet_dir]:
    if not os.path.exists(d):
      os.mkdir(d)
  
  return job_dir

def pull_data(kw_path, job_dir, method):
  info_dir = os.path.join(job_dir, 'user_information')
  tweet_dir = os.path.join(job_dir, 'user_timeline')

  if method == 'userinfo':
    os.system('python twitter_search.py get_user_information --accesstoken {} --keypath {} --kwpath {} --outdir {}'.format(ACCESS_TOKEN_PATH, KEY_PATH, kw_path, info_dir))
  
  elif method == 'pasttweets':
    # Split up user lists into N different files on-the-fly and pull past tweets
    f = open(kw_path, 'rt')
    elts = f.readlines()
    f.close()
    random.shuffle(elts)

    split_paths = []
    for i in range(N_KEYS):
      out_kw_path = '{}.{}'.format(kw_path, i)
      f = open(out_kw_path, 'wt')
      for j, elt in enumerate(elts):
        if (j % N_KEYS) == i:
          f.write(elt)
      f.close()
      split_paths.append(out_kw_path)
    
    ps = [Process(
      target=lambda apath, kpath, kwpath: os.system('python twitter_search.py user_timeline --accesstoken {} --keypath {} --kwpath {} --outdir {} --numtocache 3200'.format(
        apath, kpath, kwpath, tweet_dir)),
      args=(apath, kpath, kwpath,)) for apath, kpath, kwpath in
          zip(WORKING_ACCESS_TOKEN_PATHS, WORKING_KEY_PATHS, split_paths)]
    for p in ps:
      p.start()
    for p in ps:
      p.join()
    
    #os.system('python twitter_search.py user_timeline --accesstoken {} --keypath {} --kwpath {} --outdir {} --numtocache 3200'.format(ACCESS_TOKEN_PATH, KEY_PATH, kw_path, tweet_dir))
  else:
    raise Exception('Do not recognize method "{}"'.format(method))

for p in os.listdir(BRAND_USER_DIR):
  if not p.endswith('.txt'):
    continue
  
  m = brand_user_re.match(p)
  if m:
    job = m.group('job')
  else:
    continue
  
  parser = argparse.ArgumentParser(
    description='Pull brand data, either user information or past tweets')
  parser.add_argument('method', metavar='METHOD', choices=['userinfo',
                                                           'pasttweets'],
                      help='type of data to collect, "pasttweets" should ' +
                           'run much less frequently than user info')
  args = parser.parse_args()
  
  job_dir = make_dirs(job)
  pull_data(os.path.join(BRAND_USER_DIR, p), job_dir, args.method)
  print('Finished pulling data for "{}" users'.format(job))

for i, p in enumerate(ps):
  print(i)
  f = gzip.open(p, 'rt')
  for ln in f:
    t = json.loads(ln)
    uset.add(t['user']['id'])
  f.close()
  print(len(uset))
  
