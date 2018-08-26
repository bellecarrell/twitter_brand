#!/usr/bin/python

import pdb, gzip, os, platform, random, re, sys, time, webbrowser, json
import pickle, smtplib, traceback

import argparse

import twitterUtils as twitterUtils

from tweepy import OAuthHandler, API, TweepError
from tweepy.streaming import StreamListener, Stream

''' Change these variables to customize the script. '''
# Where the user's keys are stored.
KEY_PATH = None

# Where you want to cache the access token.
ACCESS_TOKEN_PATH = None

# Where the output files should be stored.
OUT_DIR = None

# Path to the file listing the keywords/users to follow.
KW_PATH = None

# If set to None, then queries are not constrained by location.  Set to a list
# of 2 lat/long float pairs to only select geo-coded tweets from that area.  
LOCATIONS = None

# Number of messages to store before writing them to file.
NUM_TO_CACHE = 200

# Who to e-mail when an error occurs.
ADMIN_EMAILS = ['admin.email@blah']

SCRIPT_EMAIL = 'script.email@blah'
SCRIPT_USERNAME = 'email_username'
SCRIPT_PW = 'email_pw'

# Standard number of seconds to wait between API calls.
SNOOZE = 3

# Time to wait between iterations when sampling user
# information multiple times.
ITER_SNOOZE = 10

#COLLECT_FRIENDS = True
#COLLECT_FOLLOWERS = True

METHOD_NAMES = ['keyword_stream', 'user_stream', 'network_sample',
                'get_user_ids', 'get_user_information', 'tweet_keyword_search',
                'user_timeline', 'get_retweets', 'hydrate_tweets',
                'tweet_geo_search']

def get_snooze_from_method(method, rsc):
  if method == 'network_sample':
    o = rsc['friend']['/friends/ids']
  elif method in {'get_user_ids', 'get_user_information'}:
    o = rsc['users']['/users/lookup']
  elif method == 'tweet_keyword_search':
    o = rsc['search']['/search/tweets']
  elif method == 'user_timeline':
    o = rsc['statuses']['/statuses/user_timeline']
  elif method == 'get_retweets':
    o = rsc['statuses']['/statuses/retweets/:id']
  elif method == 'hydrate_tweets':
    o = rsc['statuses']['/statuses/lookup']
  elif method == 'tweet_geo_search':
    o = rsc['geo']['/geo/search']
  else:
    print('No limit for {}'.format())
    return 1
  
  return  (60*15) // o['limit']

def emailAlert(msg):
  '''
  Sends a short email to all administrators describing what went
  wrong and that they should fix it.
  '''
  
  print('Error!', msg)
  print('Not emailing, doesn\'t work on grid')

def getTimeString():
  t = time.localtime()
  return '%d-%d-%d-%d-%d-%d' % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

def waitTillRateLimit(resetTime):
  sleepTime = resetTime - int(time.time()) + 5
  print ('About to hit rate limit')
  zzz(sleepTime)

def zzz(sleepTime):
  print(('Snoozing for %s seconds' % (sleepTime)))
  time.sleep(sleepTime)

def doOAuthDance(consumer_key, consumer_secret):
  # If we already have an access token saved, just use that
  if os.path.exists(ACCESS_TOKEN_PATH):
    accessFile = open(ACCESS_TOKEN_PATH, 'r')
    access_token_key = accessFile.readline().strip()
    access_token_secret = accessFile.readline().strip()
    accessFile.close()
    
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    return auth
  
  auth = OAuthHandler(consumer_key, consumer_secret)
  
  try:
    redirect_url = auth.get_authorization_url()
  except TweepError:
    print ('Error! Failed to get request token.')
    return None
  
  print(('Get PIN from: %s' % (redirect_url)))
  webbrowser.open(redirect_url)
  
  verifier = input('Input Twitter verifier PIN:')
  
  try:
    auth.get_access_token(verifier)
  except:
    print ('Error!  Failed to get access token.')
    return None
  
  # Save the access token for later use
  accessFile = open(ACCESS_TOKEN_PATH, 'wt')
  accessFile.write('%s\n%s' % (auth.access_token.key, auth.access_token.secret))
  accessFile.close()
  
  return auth

def getKeyAndAuth():
  '''
  Returns the proper authentication.  Relies on the keys stored in KEY_PATH.
  '''
  keyFile = open(KEY_PATH, 'rt')
  consumer_key = keyFile.readline().strip()
  consumer_secret = keyFile.readline().strip()
  keyFile.close()
  
  return doOAuthDance(consumer_key, consumer_secret)

class NetSampler:
  friendOutRe = re.compile('friendNetwork_(?P<num>\d+)\.tsv')
  followerOutRe = re.compile('followerNetwork_(?P<num>\d+)\.tsv')
  
  def __init__(self, userPath, iterate=False):
    try:
      os.mkdir(OUT_DIR)
    except:
      pass
    
    if self._getLastOutput():
      outIndex = self._getLastOutput()
    else:
      outIndex = 0
    
    self.friendOutFile = open(os.path.join(OUT_DIR,
                         'friendNetwork_%d.tsv' % (outIndex + 1)), 'a')
    self.followerOutFile = open(os.path.join(OUT_DIR,
                           'followerNetwork_%d.tsv' % (outIndex + 1)), 'a')
    
    self.iterate = iterate
    self.api = API(getKeyAndAuth())
    self.i = 0
    self.limit = self.api.rate_limit_status()
    self.runningErrCount = 0
    self.runningErrMsgs = ''
    self.usernames = self._getUsernames(userPath)
    self.already_collected = self._getCollected()
  
  def _getLastOutput(self):
    '''
    Returns the most recent follower file this script has written to. 
    '''
    followerOutputFiles = [(
                          int(NetSampler.followerOutRe.match(f).group('num')), 
                          os.path.join(OUT_DIR, f)) 
                          for f in os.listdir(OUT_DIR) if 
                          NetSampler.followerOutRe.match(f)]
    followerOutputFiles.sort()
    if not followerOutputFiles:
      return None
    return followerOutputFiles[-1][0]
  
  def _getCollected(self):
    ''' Check which users we've already collected. '''
    outputFiles = [os.path.join(OUT_DIR, f) 
                   for f in os.listdir(OUT_DIR) if 
                   (NetSampler.followerOutRe.match(f) or NetSampler.friendOutRe.match(f))]
    collected = set()
    for p in outputFiles:
      f = open(p, 'rt')
      for ln in f:
        flds = ln.strip().split()
        if len(flds) > 1:
          collected.add(ln.strip().split()[1])
      f.close()
    return collected
  
  def _getUsernames(self, userPath):
    userFile = open(userPath, 'rt')
    usernames = [line.strip().replace('@', '')
                 for line in userFile if line.strip()]
    userFile.close()
    
    return usernames
  
  def _updateErr(self, u, msg):
    '''
    Adds a new error message and emails admins if 5 errors in a row occurred.
    '''
    self.runningErrCount += 1
    
    try:
      tb = traceback.format_exc()
    except:
      tb = 0
    self.runningErrMsgs  += '%s -- %s: %s\n%s\n%s\n\n' % (
                            getTimeString(), u,
                            str(sys.exc_info()), tb, msg)
    if self.runningErrCount >= 5:
      emailAlert(self.runningErrMsgs)
      self._resetErr()
  
  def _resetErr(self):
    self.runningErrCount = 0
    self.runningErrMsgs = ''
  
  def _collectIds(self, apiFn, username, outFile):    
    outFile.write('%s\t%s' % (getTimeString(), username))
    
    idCount = 0
    next_cursor = -1
    gotIt = False
    while (next_cursor != 0) and (not gotIt):
      try:
        id_obj = apiFn(username, cursor=next_cursor)
        ids = id_obj[0]
        if 'errors' in id_obj:
          self._updateErr(username, str(id_obj))
          next_cursor = 0
        else:
          self._resetErr()
          
          next_cursor = id_obj[1][1]
          print(('Found %d ids for %s' % (len(ids), username)))
          idCount += len(ids)
        outFile.write('\t' + '\t'.join([str(id) for id in ids]))
        zzz(SNOOZE)
        #gotIt = True
      except TweepError as ex:
        #import pdb; pdb.set_trace()
        
        #if (('reason' in ex.__dict__) and
        #    ('code' in eval(ex.reason)[0]) and
        #    (eval(ex.reason)[0]['code']==88)):
        #  zzz(ITER_SNOOZE)
        #  continue
        
        zzz(SNOOZE)
        if 'Not authorized' in str(ex):
          outFile.write('\tnot_authorized')
          gotIt = True
        else:
          self._updateErr(username, str(ex))
          try:
            id_obj = apiFn(username, cursor=next_cursor)
            ids = id_obj[0]
            
            next_cursor = id_obj[1][1]
            outFile.write('\t' + '\t'.join([str(id) for id in ids]))
            idCount += len(ids)
            self._resetErr()
            zzz(SNOOZE)
            #gotIt = True
          except Exception as ex:
            outFile.write('\tmissing')
            self._updateErr(username, str(ex))
            next_cursor = 0
            print((username + ':', ex))
            gotIt = True
      print(('Current ID total:', idCount))
    outFile.write('\n')
  
  def sampleNetwork(self):
    print(( 'Collecting networks for %d usernames' % (len(self.usernames))))
    
    go = True
    while go:
      go = self.iterate
      for username in self.usernames:
        self.i += 1
        if not self.i%50:
          print('%d iterations - %s' % (self.i, str(self.limit)))
        
        #if username in self.already_collected:
        #  print ('Skipping %s, already collected' % (username))
        #  continue
        
        try:
          self._collectIds(self.api.friends_ids,
                           username, self.friendOutFile)
          self._collectIds(self.api.followers_ids,
                           username, self.followerOutFile)
          self.limit = self.api.rate_limit_status()
        #except urllib.error.URLError as ex:
        #  print((username + ':', ex))
        #  self._updateErr(username, str(ex))
        #  emailAlert(runningErrMsgs + '\nPress enter to reconnect!!!')
        #  zzz(SNOOZE)
        #  input('Press enter to reconnect')
        #  self.api = API(getKeyAndAuth())
        except TweepError as ex:
          print((username + ':', ex))
          self._updateErr(username, str(ex))
          zzz(SNOOZE)
          try:
            self.limit = self.api.rate_limit_status()
          except:
            pass
      if go:
        zzz(ITER_SNOOZE)

class TermListener (StreamListener):
  '''
  Listener for the streaming API (for keyword/user streams).  Dumps tweets
  in batches of NUM_TO_CACHE per file.
  '''
  
  def __init__(self, numCached):
    super(TermListener, self).__init__()
    self.tweets = []
    self.startTime = getTimeString()
    self.i = 0
    self.numCached = numCached
    self.time = time.time()
  
  def on_status(self, status):
    self.tweets.append(status)
    self.i += 1
    
    if len(self.tweets) >= self.numCached:
      self._dumpTweets()
  
  def on_error(self, status_code):
    print((getTimeString(), 'Error:', status_code))
    return True
  
  def on_timeout(self):
    print((getTimeString(), 'Bored. . . . . .'))
    
    # If we haven't gotten a tweet in an hour, then dump what we have so far.
    if self.tweets and (time.time() - self.time >= 3600):
      self._dumpTweets()
  
  def _dumpTweets(self):
    print((getTimeString(), 'Number of tweets:', self.i))
    currTime = getTimeString()
    with gzip.open(os.path.join(OUT_DIR,
                                        '{}_to_{}.json.gz'.format(
                                          self.startTime, currTime
                                        )), 'wt') as out_file:
      for t in self.tweets:
        out_file.write(json.dumps(t._json) + '\n')
    
    self.startTime = currTime
    self.time = time.time()
    self.tweets = []
  
  def on_limit(self, track):
    print(( 'Limited', track))
    return True

def getIds(usernamePath):
  f = open(usernamePath)
  users = [line.strip().replace('@', '') for line in f if line.strip()]
  f.close()
  
  return [(o.screen_name, o.name, i) for i, o in getUserData(users)]

def writeIds(username_path, out_path):
  ret_list = getIds(username_path)
  
  outFile = open(out_path, 'wt')
  outFile.write('\n'.join(['{}\t{}\t{}'.format(*vs) for vs in userIds]))
  outFile.close()

def _mkBunches(xs, step=100):
  bunches = []
  for i in range(0, len(xs), step):
    bunches.append(xs[i:(i+step)])
  return bunches

def getUserData(users):
  '''
  Collects user data for all users in a set, and dumps them to file.
  Assumes that usernames are in the file, not their twitter IDs.  Looks
  users up 100 at a time.
  '''
  
  uIds = []
  uNames = []
  for u in users:
    try:
      uIds.append(int(u))
    except:
      uNames.append(u)
  
  api = API(getKeyAndAuth())
  
  userInfo = []
  
  idBunches = _mkBunches(uIds)
  nameBunches = _mkBunches(uNames)
  
  i = 0
  for ids in idBunches:
    gotIt = False
    
    while not gotIt:
      try:
        userObjs = api.lookup_users(user_ids=ids)
        gotIt = True
        i += len(userObjs)
        print(( 'Collected info for %d users' % (i)))
        userInfo.extend([(o.id, o) for o in userObjs])
      except Exception as e:
        try:
          if (('reason' in e.__dict__) and
              ('code' in eval(e.reason)[0]) and
              (eval(e.reason)[0]['code']==88)):
            print (e)
            print(('Rate limit exceeded: ', ids[0]))
            zzz(ITER_SNOOZE)
          else:
            print (e)
            print(('Cannot find IDs starting at:', ids[0]))
            gotIt = True
        except:
          print(e)
          print('Cannot find IDs starting at:', ids[0])
          gotIt = True
    zzz(SNOOZE)
  
  for names in nameBunches:
    gotIt = False
    
    while not gotIt:
      try:
        userObjs = api.lookup_users(screen_names=names)
        gotIt = True
        i += len(userObjs)
        print(getTimeString(), 'Collected info for %d users' % (i))
        userInfo.extend([(o.id, o) for o in userObjs])
      except Exception as e:
        if (('reason' in e.__dict__) and
            ('code' in eval(e.reason)[0]) and
            (eval(e.reason)[0]['code']==88)):
          print(e)
          print('Rate limit exceeded: ', statusId)
          zzz(ITER_SNOOZE)
        else:
          print(e)
          print('Cannot find:', statusId)
          gotIt = True
    zzz(SNOOZE)
  
  return userInfo

def kwSearch(kwPath, since_id):
  kwFile = open(kwPath, 'rt')
  kws = [line.strip().replace('@', '') for line in kwFile if line.strip()]
  kwFile.close()
  
  try:
    os.mkdir(outDir)
  except:
    pass
  
  api = API(getKeyAndAuth())
  
  for kw in kws:
    kwResults = []
    tmp_since_id = since_id
    print('Searching for %s tweets' % (kw))
    nextPage = True
    p = 1
    while nextPage:
      try:
        searchRes = api.search([kw], since_id=tmp_since_id, rpp=100, page=p)
        print(getTimeString(), 'Found %d %s tweets' % (len(searchRes), kw))
        kwResults.extend(searchRes)
        zzz(SNOOZE)
        
        if searchRes.next_page:
          nextPage = True
          p += 1
          print(getTimeString(), 'New page: %s' % searchRes.next_page)
        else:
          nextPage = False
      except Exception as ex:
        print(ex)
        break

    
    with gzip.open(os.path.join(OUT_DIR,
                                '{}.json.gz'.format(kw)), 'wt') as out_file:
      for t in kwResults:
        out_file.write(json.dumps(t._json).strip() + '\n')

class Collect:
  '''
  Used to collect past tweets that users have made.
  '''
  
  def __init__(self):
    pass
  
  def getPastTweets(self):
    api = API(getKeyAndAuth())
    
    userFile = open(KW_PATH, 'r')
    totalUsers = [line.strip() for line in userFile if line.strip()]
    userFile.close()
    
    i = 0
    currentPercentage = 0
    total = len(totalUsers)
    
    maxBatchedUsers = 50
    
    batchedStatuses = []
    
    start_user = str(totalUsers[0])
    
    for user in totalUsers:
      i += 1
      
      #if os.path.exists(os.path.join(OUT_DIR, '%s_statuses.pickle' % (user))) or os.path.exists(os.path.join(OUT_DIR, '%s_statuses.pickle.gz' % (user))):
      #  continue
      
      if not i%50:
        limit = api.rate_limit_status()
        print('\n%d - %s' % (i, str(limit)))
      
      if (i*100)/total > currentPercentage:
        currentPercentage = (i*100)/total
        print('%d%% done!' % (currentPercentage))
      
      tryAgain = True
      
      userTimeline = []
      userAllStatuses = []
      
      if start_user == None:
        start_user = user
      
      maxId = 0
      while (userTimeline or tryAgain) and len(userAllStatuses) < NUM_TO_CACHE:
        try:
          print('Searching for statuses from', user, 'of max ID', maxId, 'current count', len(userAllStatuses))
          tryAgain = False
          if maxId > 0:
            try:
              u = int(user)
              userTimeline = api.user_timeline(user_id=u,  max_id=maxId, count=200, include_rts=True)
            except:
              userTimeline = api.user_timeline(screen_name=user, max_id=maxId, count=200, include_rts=True)
          else:
            try:
              u = int(user)
              userTimeline = api.user_timeline(user_id=u, count=200, include_rts=True)
            except:
              userTimeline = api.user_timeline(screen_name=user, count=200, include_rts=True)
          userAllStatuses.extend(userTimeline)
          maxId = min([s.id for s in userTimeline])-1 if userTimeline else 0
          zzz(SNOOZE)
          
          #try:
          #  limit = api.rate_limit_status()
          #except:
          #  print('Trouble getting limit. . . OH WELL')
        except TweepError as e:
          try:
            raise twitterUtils.getRefinedTweepError(e)
          except twitterUtils.NotModTweepError as e2:
            print('Error code %d: No data to return. . . boo' % (e2.response))
          
          except twitterUtils.RateLimitTweepError as e2:
            print('Error code %d: Rate limit hit, snoozing for %s seconds' % (e2.response, str(limit['reset_time_in_seconds'])))
            waitTillRateLimit(limit['reset_time_in_seconds'])
          
          except twitterUtils.UnauthTweepError as e2:
            print('Error code %d: This user has a private account'  % (e2.response))
            
          except twitterUtils.ForbiddenTweepError as e2:
            print('Error code %d: Update limit hit (I should not be happening!!)' % (e2.response))
          
          except twitterUtils.NotFoundTweepError as e2:
            print('Error code %d: %d does not exist.'  % (e2.response, user))
          
          except twitterUtils.NotAcceptableTweepError as e2:
            print('Invalid search format'  % (e2.response))
          
          except twitterUtils.ServerTweepError as e2:
            print('Error code %d: Server\'s having some troubles, try again in a bit' % (e2.response))
            zzz(SNOOZE)
            tryAgain = True
            continue
          except TweepError as e2:
            print('Unknown TweepError: %s' % (e2.reason))
            zzz(2*SNOOZE)
        except Exception as e:
          print(e)
          zzz(2*SNOOZE)
        
        batchedStatuses += userAllStatuses
        if (i % maxBatchedUsers) == 0:
          with gzip.open(os.path.join(OUT_DIR,
                                      '{}_to_{}.statuses.json.gz'.format(start_user, user)),
                         'wt') as out_file:
            curr_time = time.time()
            for t in batchedStatuses:
              tdict = t._json
              tdict['collected_at_unix_timestamp'] = curr_time
              out_file.write(json.dumps(tdict).strip() + '\n')
          
          print('%s to %s\tdumped\n' % (start_user, user))
          
          start_user = user
          batchedStatuses = []
        
      with gzip.open(os.path.join(OUT_DIR,
                                  '{}_to_{}.statuses.json.gz'.format(start_user,
                                                                  user)), 'wt') as out_file:
        curr_time = time.time()
        for t in batchedStatuses:
          tdict = t._json
          tdict['collected_at_unix_timestamp'] = curr_time
          out_file.write(json.dumps(tdict).strip() + '\n')
      print('%s to %s\tdumped\n' % (start_user, user))

class KeywordStreamer:
  '''
  Streams in tweets matching a set of terms/from a set of user streams.
  '''
  
  def __init__(self, termPath):
    try:
      os.mkdir(OUT_DIR)
    except:
      pass
    
    kwFile = open(termPath, 'r')
    self.kws = [line.strip() for line in kwFile if line.strip()]
    kwFile.close()
    
    self.stream = Stream(getKeyAndAuth(), TermListener(NUM_TO_CACHE))
  
  def streamTweets(self, searchForTerms=True):
    goForIt = True
    while goForIt:
      try:
        goForIt = False
        if searchForTerms:
          self.stream.filter(track=self.kws, locations=LOCATIONS)
        else:
          self.stream.filter(follow=[int(k) for k in self.kws],
                             locations=LOCATIONS)
      #except IncompleteRead as ex:
      #  try:
      #    tb = traceback.format_exc()
      #  except:
      #    tb = ''
      #  emailAlert("%s: %s\n%s\n\n" % (getTimeString(),
      #             str(sys.exc_info()) + "\nStill ticking!", tb))
      #  zzz(SNOOZE)
      #  goForIt = True
      except:
        try:
          tb = traceback.format_exc()
        except:
          tb = ''
        emailAlert('%s: %s\n%s\n\n' % (getTimeString(),
                                       str(sys.exc_info()), tb))
        zzz(SNOOZE)
        goForIt = True

def getStatuses(statusIdPath, retrieve_by_batch=True):
  '''
  Retrieve hydrated tweets from tweet ID.
  '''
  
  idFile = open(statusIdPath, 'rt')
  ids = [int(line.strip()) for line in idFile if line.strip()]
  idFile.close()
  
  api = API(getKeyAndAuth(), api_root='/1.1')
  
  with gzip.open(os.path.join(OUT_DIR, 'statuses.{}_{}.json.gz'.format(ids[0],
                                                                       ids[1])),
                 'wt') as out_file:
    if retrieve_by_batch:
      id_batches = [ids[idx:(idx+100)] for idx
                    in range(0, len(ids)+100, 100) if idx < len(ids)]
      
      for ids in id_batches:
        sobjs = api.statuses_lookup(ids)
        for o in sobjs:
          out_file.write(json.dumps(o._json), 'wt')
        
        print('Grabbed ids {} to {} -- {} found'.format(ids[0],
                                                        ids[1],
                                                        len(sobjs)))
        zzz(SNOOZE)
    else:
      for statusId in ids:
        try:
          status = api.get_status(statusId)
        except Exception as e:
          print('Error getting ID %d -- ' % (statusId), e)
          
          out_file.write(json.dumps({'error':str(e),
                                     'id':statusId}).strip() + '\n')
          zzz(SNOOZE)
        
          continue
        
        out_file.write(json.dumps({'error':str(e),
                                   'id':statusId}).strip() + '\n')
        zzz(SNOOZE)

def getRTs(statusIdPath):
  '''
  Runs through a list of status IDs and grabs up to 100 of their RTs.
  '''
  
  idFile = open(statusIdPath, 'rt')
  ids = [int(line.strip()) for line in idFile if line.strip()]
  idFile.close()
  
  api = API(getKeyAndAuth())
  
  userInfo = []
  
  for statusId in ids:
    gotIt = False
    while not gotIt:
      try:
        if os.path.exists(os.path.join(OUT_DIR, '%d.rts.json.gz' % (statusId))):
          gotIt = True
          continue
        
        rts = api.retweets(id=statusId, count=100)
        gotIt = True
        with gzip.open(os.path.join(OUT_DIR,
                                         '%d.rts.json.gz' % (statusId)), 'wt') as out_file:
          for rt in rts:
            out_file.write(json.dumps(rt._json).strip() + '\n')
        print(getTimeString(), ': %d RTs for %d' % (len(rts), statusId))
      except Exception as e:
        if (('reason' in e.__dict__) and
            ('code' in eval(e.reason)[0]) and
            (eval(e.reason)[0]['code']==88)):
          print(e)
          print('Rate limit exceeded: ', statusId)
          zzz(ITER_SNOOZE)
        else:
          print(e)
          print('Cannot find:', statusId)
          with gzip.open(os.path.join(OUT_DIR,
                                      '%d.rts.json.gz' % (statusId)), 'wt') as out_file:
            out_file.write(json.dumps({'id':statusId, 'error':str(e)}).strip()
                           + '\n')
          
          gotIt = True
      zzz(SNOOZE)

def getGeoLocations(geoQueryPath):
  '''
  For a set of latlongs or location names, attempt to find the Twitter
  locations they correspond to.
  '''
  
  f = open(geoQueryPath)
  queries = [line.strip() for line in f]
  f.close()
  
  latLngRe = re.compile('-?\d+\.\d+,-?\d+\.\d+')
  
  try:
    os.mkdir(OUT_DIR)
  except:
    pass
  
  api = API(getKeyAndAuth())
  
  # If these are latlongs, cast them as such.  Otherwise treated as strings.
  if latLngRe.match(queries[0]):
    queries = [(float(q.split(',')[0]),
                float(q.split(',')[1])) for q in queries]
    
    for lat, lng in queries:
      
      notFinished = True
      while notFinished:
        try:
          places = api.reverse_geocode(lat=lat, int=lng, accuracy='10m')
        except Exception as ex:
          print('Exception with <%f, %f>:' % (lat,lng), ex)
          zzz(SNOOZE)
          continue
        
        with open(os.path.join(OUT_DIR, '%f.%f.json' % (lat, lng)), 'w') as out_file:
          for p in places:
            out_file.write(json.dumps(p._json).strip() + '\n')
        
        notFinished = False
        
        print('Wrote out <%f, %f>' % (lat, lng))
        zzz(SNOOZE)
  else:
    for q in queries:
      notFinished = True
      outPath = os.path.join(OUT_DIR, '%s.json' %
                             (q.lower().replace(',', '_')
                                       .replace('"', '')
                                       .replace('/', '_')))
      
      if os.path.exists(outPath):
        continue
      
      while notFinished:
        try:
          places = api.geo_search(query=q)
        except Exception as ex:
          print('Exception with <%s>:' % (q), ex)
          zzz(SNOOZE)
          continue
        
        with open(outPath, 'wt') as out_file:
          for p in places:
            out_file.write(json.dumps(p._json).strip() + '\n')
        
        notFinished = False
        print('Wrote out <%s>' % (q))
        zzz(SNOOZE)

def _parseLocString(locStr):
  latLongs = locStr.replace(' ', '').split(',')
  assert (len(latLongs) % 4 == 0) and (len(latLongs) > 0)
  return [float(v) for v in latLongs]

def collect_user_information(args):
  usernameFile = open(args.kwPath, 'rt')
  users = [line.strip().replace('@', '') for line in usernameFile
           if line.strip()]
  usernameFile.close()
  
  bunches = _mkBunches(users)
  
  todo = True
  while todo or args.iter:
    todo = False
    for bunch in bunches:
      infos = getUserData(bunch)
      
      with gzip.open(os.path.join(OUT_DIR,
                                  'userInfo_{}_to_{}.{}.json.gz'.format(
                                    bunch[0], bunch[-1], getTimeString())),
                     'wt') as out_file:
        
        for uid, uobj in infos:
          user_dict = uobj._json
          user_dict['collected_at_unix_timestamp'] = time.time()
          out_file.write(json.dumps(user_dict).strip() + '\n')
    if args.iter:
      zzz(ITER_SNOOZE)

def branch_on_method(args):
  global SNOOZE
  
  api = API(getKeyAndAuth())
  limit = api.rate_limit_status()

  optimal_snooze = get_snooze_from_method(args.method, limit['resources'])
  print('Setting snooze time to {}'.format(optimal_snooze))
  SNOOZE = optimal_snooze
  
  if args.method in {'keyword_stream', 'user_stream'}:
    streamer = KeywordStreamer(args.kwPath)
    streamer.streamTweets( args.method == 'keyword_stream' )
  elif args.method == 'network_sample':
    netSampler = NetSampler(args.kwPath, iterate=args.iter)
    netSampler.sampleNetwork()
  elif args.method == 'get_user_ids':
    out_path = os.path.join(OUT_DIR, 'user_ids.txt')
    idx = 1
    while os.path.exists(out_path):
      out_path = os.path.join(OUT_DIR, 'user_ids.{}.txt'.format(idx))
    
    writeIds(args.kwPath, out_path)
  elif args.method == 'get_user_information':
    collect_user_information(args)
  elif args.method == 'tweet_keyword_search':
    if not args.sinceId:
      print('Must input "since status ID" when searching.')
    else:
      kwSearch(args.kwPath, args.sinceId)
  elif args.method == 'user_timeline':
    statusCollecter = Collect()
    statusCollecter.getPastTweets()
  elif args.method == 'get_retweets':
    getRTs(args.kwPath)
  elif args.method == 'hydrate_tweets':
    getStatuses(args.kwPath)
  elif args.method == 'tweet_geo_search':
    getGeoLocations(args.kwPath)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pull Twitter data')
  parser.add_argument('method', metavar='METHOD', choices=METHOD_NAMES,
                      help='function to call')
  parser.add_argument('--keypath', dest='keyPath',
                    help='specifies KEY_PATH, containing the consumer' +
                         ' key and secret to use', metavar='KEY_PATH',
                    required=True)
  parser.add_argument('-a', '--accesstoken', dest='accessToken',
                    help='specifies ACCESS_TOKEN_PATH, the file where' +
                         ' the access token will be saved/used in the future.',
                    metavar='ACCESS_TOKEN_PATH',
                    required=True)
  parser.add_argument('--sinceid', dest='sinceId', type=int,
                    help='used when searching for tweets with keywords.  The' +
                         ' minimum status ID to start searching from.',
                    metavar='SINCE_ID')
  parser.add_argument('--kwpath', dest='kwPath',
                    help='specifies KW_PATH, containing all the terms/users' +
                         ' to search for.', metavar='KW_PATH', required=True)
  parser.add_argument('-o', '--outdir', dest='outDir',
                    help='specifies the OUT_DIR where files will be dumped',
                    metavar='OUT_DIR', required=True)
  parser.add_argument('--numtocache', dest='numToCache', type=int, default=100,
                    help='when streaming tweets, will wait for NUM_TO_CACHE' +
                         ' messages before writing to disk.  When collecting' +
                         ' past user tweets, will find up to NUM_TO_CACHE' +
                         ' most recent tweets.',
                    metavar='NUM_TO_CACHE')
  parser.add_argument('--iter', dest='iter', action='store_true', default=False,
                    help='this flag causes the script to continuously sample' +
                    ' data when grabbing user information or networks.')
  parser.add_argument('-l', '--location', dest='location',
                    help='a comma-separated list of four lat-long floats.' +
                         '  This retrieves tweets from these' +
                         ' geo-coded regions. Specify the SW point first.',
                    metavar='LOCATION')
  parser.add_argument('--snooze',
                    dest='snooze', default=SNOOZE, type=int,
                    help='Seconds to wait between API calls.')
  parser.add_argument('--itersnooze', type=int,
                    dest='itersnooze', default=ITER_SNOOZE,
                    help='Seconds to wait between iterations over users (tracking network/user information)')
  args = parser.parse_args()
  
  if args.keyPath:
    KEY_PATH = args.keyPath
  if args.accessToken:
    ACCESS_TOKEN_PATH = args.accessToken
  if args.kwPath:
    KW_PATH = args.kwPath
  if args.outDir:
    OUT_DIR = args.outDir
  if args.numToCache:
    NUM_TO_CACHE = int(args.numToCache)
  if args.location:
    LOCATIONS = _parseLocString(args.location)

  SNOOZE = args.snooze
  ITER_SNOOZE = args.itersnooze
  
  if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
  
  try:
    branch_on_method(args)
  except Exception as ex:
    try:
      tb = str(traceback.format_exc())
    except:
      tb = ''
    emailAlert('%s\n%s\n%s\n\n' % (str(sys.exc_info()), tb, str(ex)))
  
