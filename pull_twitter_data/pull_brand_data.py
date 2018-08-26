'''
Pulls data for brand-like, self-promoting users.

Adrian Benton
8/25/2018
'''

import os, re, time

TOP_DIR = '/exp/abenton/twitter_brand_data/'
#if not os.path.exists(TOP_DIR): os.mkdir(TOP_DIR)

KEY = 'dl1'
KEY_PATH = '/exp/abenton/twitter_collection/keys/{}_keys.txt'.format(KEY)
ACCESS_TOKEN_PATH = '/exp/abenton/twitter_collection/access_tokens/access_token_{}.txt'.format(KEY)

BRAND_USER_DIR = '/exp/abenton/twitter_collection/brand_lists/'

brand_user_re = re.compile('(?P<job>\w+)\.users\.txt')

def getTimeString():
  t = time.localtime()
  return '%d-%d-%d-%d-%d-%d' % (t.tm_year, t.tm_mon,
                                t.tm_mday, t.tm_hour,
                                t.tm_min, t.tm_sec)

def make_dirs(job):
  job_dir = os.path.join(TOP_DIR, job)
  if not os.path.exists(job_dir):
    os.mkdir(job_dir)
  
  time_stamp = time.time()
  time_dir = os.path.join(job_dir, '{}'.format(time_stamp))
  if not os.path.exists(time_dir):
    os.mkdir(time_dir)
  
  info_dir = os.path.join(time_dir, 'user_information')
  tweet_dir = os.path.join(time_dir, 'user_timeline')
  #network_dir = os.path.join(time_dir, 'networks')

  for d in [info_dir, tweet_dir]:
    if not os.path.exists(d):
      os.mkdir(d)

  return time_dir

def pull_data(kw_path, time_dir):
  info_dir = os.path.join(time_dir, 'user_information')
  tweet_dir = os.path.join(time_dir, 'user_timeline')

  os.system('python twitter_search.py get_user_information --accesstoken {} --keypath {} --kwpath {} --outdir {}'.format(ACCESS_TOKEN_PATH, KEY_PATH, kw_path, info_dir))
  
  os.system('python twitter_search.py user_timeline --accesstoken {} --keypath {} --kwpath {} --outdir {} --numtocache 3200'.format(ACCESS_TOKEN_PATH, KEY_PATH, kw_path, tweet_dir))

for p in os.listdir(BRAND_USER_DIR):
  m = brand_user_re.match(p)
  if m:
    job = m.group('job')
  else:
    continue
  
  time_dir = make_dirs(job)
  pull_data(os.path.join(BRAND_USER_DIR, p), time_dir)
  print('Finished pulling data for "{}" users'.format(job))
