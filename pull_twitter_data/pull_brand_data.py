'''
Pulls data for brand-like, self-promoting users.

Adrian Benton
8/25/2018
'''
#!/usr/bin/env python

import os, re, time

TOP_DIR = '/exp/acarrell/twitter_brand/user_infos/'
#if not os.path.exists(TOP_DIR): os.mkdir(TOP_DIR)

KEY = 'dl1'
KEY_PATH = '/exp/abenton/twitter_collection/keys/{}_keys.txt'.format(KEY)
ACCESS_TOKEN_PATH = '/exp/abenton/twitter_collection/access_tokens/access_token_{}.txt'.format(KEY)

BRAND_USER_DIR = '/exp/acarrell/twitter_brand/blogger_2018/17_allids/'

brand_user_re = re.compile('(?P<job>\w+)\.users\.txt')

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

def pull_data(kw_path, job_dir):
  info_dir = os.path.join(job_dir, 'user_information')
  tweet_dir = os.path.join(job_dir, 'user_timeline')
  
  os.system('python twitter_search.py get_user_information --accesstoken {} --keypath {} --kwpath {} --outdir {}'.format(ACCESS_TOKEN_PATH, KEY_PATH, kw_path, info_dir))
  
  #os.system('python twitter_search.py user_timeline --accesstoken {} --keypath {} --kwpath {} --outdir {} --numtocache 200'.format(ACCESS_TOKEN_PATH, KEY_PATH, kw_path, tweet_dir))

for p in os.listdir(BRAND_USER_DIR):
  m = brand_user_re.match(p)
  if m:
    job = m.group('job')
  else:
    continue
  
  job_dir = make_dirs(job)
  pull_data(os.path.join(BRAND_USER_DIR, p), job_dir)
  print('Finished pulling data for "{}" users'.format(job))
