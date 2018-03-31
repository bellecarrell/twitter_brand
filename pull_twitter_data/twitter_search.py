#!/usr/bin/python

import pdb, gzip, os, platform, random, re, sys, time, webbrowser, json
import pickle, smtplib, traceback

from optparse import OptionParser

import twitterUtils

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
ADMIN_EMAILS = ['foo.bar@blah']

SCRIPT_EMAIL = 'tweet.scooper@gmail.com'
SCRIPT_USERNAME = 'tweet.scooper'
SCRIPT_PW = 'YTtDZB3j'

# Standard number of seconds to wait between API calls.
SNOOZE = 3

# Time to wait between iterations when sampling user
# information multiple times.
ITER_SNOOZE = 10

#COLLECT_FRIENDS = True
#COLLECT_FOLLOWERS = True

def emailAlert(msg):
  '''
  Sends a short email to all administrators describing what went
  wrong and that they should fix it.
  '''
  
  print('Error!', msg)
  print('Not emailing, doesn\'t work on grid')
  
  '''
  print(( 'Sending message:', msg))
  mailserver = smtplib.SMTP('smtp.gmail.com')
  mailserver.ehlo()
  mailserver.starttls()
  mailserver.login(SCRIPT_USERNAME, SCRIPT_PW)
  try:
    msg = 'Trouble on %s\n\n%s' % (os.uname(), msg)
  except:
    msg = 'Trouble on %s\n\n%s' % (platform.platform(), msg)
  for admin in ADMIN_EMAILS:
    headers = "From: %s\nTo: %s\nSubject: Trouble's afoot...\n\n" % (
              SCRIPT_EMAIL, admin)
    message = headers + msg
    mailserver.sendmail(SCRIPT_EMAIL, admin, message)
  mailserver.close()
  '''

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
  accessFile = open(ACCESS_TOKEN_PATH, 'w')
  accessFile.write('%s\n%s' % (auth.access_token.key, auth.access_token.secret))
  accessFile.close()
  
  return auth

def getKeyAndAuth():
  '''
  Returns the proper authentication.  Relies on the keys stored in KEY_PATH.
  '''
  keyFile = open(KEY_PATH, 'r')
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
      f = open(p)
      for ln in f:
        flds = ln.strip().split()
        if len(flds) > 1:
          collected.add(ln.strip().split()[1])
      f.close()
    return collected
  
  def _getUsernames(self, userPath):
    userFile = open(userPath, 'r')
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
          #self._collectIds(self.api.friends_ids,
          #                 username, self.friendOutFile)
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
    pickleFile = gzip.open(os.path.join(OUT_DIR, '%s_to_%s.pickle.gz' %
                          (self.startTime, currTime)), 'wb')
    pickle.dump(self.tweets, pickleFile)
    pickleFile.close()
    
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
  kwFile = open(kwPath, 'r')
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
    outFile = gzip.open(os.path.join(OUT_DIR, kw + '.pickle.gz'), 'wb')
    pickle.dump(kwResults, outFile)
    outFile.close()

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
    
    maxBatchedUsers = 500
    
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
        outFile = gzip.open(os.path.join(OUT_DIR,
                                         '%s_%s_statuses.pickle.gz' % (start_user, user)), 'wb')
        pickle.dump(batchedStatuses, outFile)
        outFile.close()
        print('%s to %s\tdumped\n' % (start_user, user))
        
        start_user = None
        batchedStatuses = []
    
    outFile = gzip.open(os.path.join(OUT_DIR, '%s_%s_statuses.pickle.gz' %
                                     (start_user, user)), 'wb')
    pickle.dump(batchedStatuses, outFile)
    outFile.close()
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

def getStatuses(statusIdPath):
  '''
  Retrieve hydrated tweets from tweet ID.
  '''
  
  idFile = open(statusIdPath, 'r')
  ids = [int(line.strip()) for line in idFile if line.strip()]
  idFile.close()
  
  api = API(getKeyAndAuth(), api_root='/1.1')
  
  batch = []
  
  for statusId in ids:
    
    try:
      status = api.get_status(statusId)
    except Exception as e:
      print('Error getting ID %d -- ' % (statusId), e)
      batch.append((statusId, str(e)))
      zzz(SNOOZE)
      
      continue
    
    batch.append(status)
    
    print('Grabbed status ID: %d' % (statusId))
    #print batch[-1]
    
    if len(batch) >= 10:
      outFile = gzip.open(os.path.join(OUT_DIR,
                                       '%d.pickle.gz' % (statusId)), 'w')
      pickle.dump(batch, outFile)
      outFile.close()
      
      batch = []
    
    zzz(SNOOZE)
  
  if batch:
    outFile = gzip.open(os.path.join(OUT_DIR,
                                     '%d.pickle.gz' % (statusId)), 'w')
    pickle.dump(batch, outFile)
    outFile.close()

def getRTs(statusIdPath):
  '''
  Runs through a list of status IDs and grabs up to 100 of their RTs.
  '''
  
  idFile = open(statusIdPath, 'r')
  ids = [int(line.strip()) for line in idFile if line.strip()]
  idFile.close()
  
  api = API(getKeyAndAuth())
  
  userInfo = []
  
  for statusId in ids:
    gotIt = False
    while not gotIt:
      try:
        if os.path.exists(os.path.join(OUT_DIR, '%d.pickle.gz' % (statusId))):
          gotIt = True
          continue
        
        rts = api.retweets(id=statusId, count=100)
        gotIt = True
        outFile = gzip.open(os.path.join(OUT_DIR,
                                         '%d.pickle.gz' % (statusId)), 'wb')
        pickle.dump(rts, outFile)
        outFile.close()
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
        
        outFile = open(os.path.join(OUT_DIR, '%f.%f.pickle' % (lat, lng)), 'w')
        pickle.dump(places, outFile)
        outFile.close()
        
        notFinished = False
        
        print('Wrote out <%f, %f>' % (lat, lng))
        zzz(SNOOZE)
  else:
    for q in queries:
      notFinished = True
      outPath = os.path.join(OUT_DIR, '%s.pickle' %
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
        
        outFile = open(outPath, 'w')
        pickle.dump(places, outFile)
        outFile.close()
        
        notFinished = False
        print('Wrote out <%s>' % (q))
        zzz(SNOOZE)

def _parseLocString(locStr):
  latLongs = locStr.replace(' ', '').split(',')
  assert (len(latLongs) % 4 == 0) and (len(latLongs) > 0)
  return [float(v) for v in latLongs]

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('--keypath', dest='keyPath',
                    help='specifies KEY_PATH, containing the consumer' +
                         ' key and secret to use', metavar='KEY_PATH')
  parser.add_option('-a', '--accesstoken', dest='accessToken',
                    help='specifies ACCESS_TOKEN_PATH, the file where' +
                         ' the access token will be saved/used in the future.',
                         metavar='ACCESS_TOKEN_PATH')
  parser.add_option('--sinceid', dest='sinceId',
                    help='used when searching for tweets with keywords.  The' +
                         ' minimum status ID to start searching from.',
                    metavar='SINCE_ID')
  parser.add_option('--kwpath', dest='kwPath',
                    help='specifies KW_PATH, containing all the terms/users' +
                         ' to search for.', metavar='KW_PATH')
  parser.add_option('-o', '--outdir', dest='outDir',
                    help='specifies the OUT_DIR where files will be dumped',
                    metavar='OUT_DIR')
  parser.add_option('--numtocache', dest='numToCache', type='int', default=100,
                    help='when streaming tweets, will wait for NUM_TO_CACHE' +
                         ' messages before writing to disk.  When collecting' +
                         ' past user tweets, will find up to NUM_TO_CACHE' +
                         ' most recent tweets.',
                    metavar='NUM_TO_CACHE')
  parser.add_option('--iter', dest='iter', action='store_true', default=False,
                    help='this flag causes the script to continuously sample' +
                    ' data when grabbing user information or networks.')
  parser.add_option('-l', '--location', dest='location',
                    help='a comma-separated list of four lat-long floats.' +
                         '  This retrieves tweets from these' +
                         ' geo-coded regions. Specify the SW point first.',
                    metavar='LOCATION')
  parser.add_option('-k', '--kwstream', action='store_true',
                    dest='kw_search', default=False,
                    help='search for tweets mentioning a keyword')
  parser.add_option('-u', '--userfollow', action='store_true',
                    dest='user_follow', default=False,
                    help='follow a set of users')
  parser.add_option('-n', '--netsample', action='store_true',
                    dest='net_sample', default=False,
                    help='sample networks for a collection of users')
  parser.add_option('-g', '--getids', action='store_true',
                    dest='get_ids', default=False,
                    help='gets and dumps the IDs for a list of usernames')
  parser.add_option('-i', '--info', action='store_true', dest='get_info',
                    default=False,
                    help='gets and dumps the user info for a list of users')
  parser.add_option('-s', '--search', action='store_true', dest='search_kws',
                    default=False,
                    help='''searches for tweets containing keywords.  Must
                            be used in conjunction with --sinceid.
                         ''')
  parser.add_option('-p', '--pasttweets', action='store_true',
                           dest='retrieve_user_timeline', default=False,
                           help='gets and dumps all tweets ' + 
                                   'made by a list of users')
  parser.add_option('-r', '--retweets', action='store_true',
                    dest='retrieve_retweets', default=False,
                    help='gets and dumps all retweets for a list of ' +
                         'status IDs')
  parser.add_option('--statuses', action='store_true',
                    dest='retrieve_statuses', default=False,
                    help='gets actual statuses for a set of IDs')
  parser.add_option('--geosearch', action='store_true',
                    dest='geo_search', default=False,
                    help='Gets locations from query string')
  (options, args) = parser.parse_args()
  
  if options.keyPath:
    KEY_PATH = options.keyPath
  if options.accessToken:
    ACCESS_TOKEN_PATH = options.accessToken
  if options.kwPath:
    KW_PATH = options.kwPath
  if options.outDir:
    OUT_DIR = options.outDir
  if options.numToCache:
    NUM_TO_CACHE = int(options.numToCache)
  if options.location:
    LOCATIONS = _parseLocString(options.location)
  
  try:
    os.mkdir(OUT_DIR)
  except:
    pass
  
  try:
    if options.net_sample:
      netSampler = NetSampler(KW_PATH, iterate=options.iter)
      netSampler.sampleNetwork()
    elif options.user_follow:
      streamer = KeywordStreamer(KW_PATH)
      streamer.streamTweets(False)
    elif options.kw_search:
      streamer = KeywordStreamer(KW_PATH)
      streamer.streamTweets(True)
    elif options.get_ids:
      userIds = getIds(KW_PATH)
      
      #import pdb; pdb.set_trace()
      outFile = open(os.path.join(OUT_DIR, 'user_ids.txt'), 'w')
      outFile.write('\n'.join(['\t'.join([str(v) for v in p])
                                          for p in userIds]))
      outFile.close()
    elif options.get_info:
      usernameFile = open(KW_PATH, 'r')
      users = [line.strip().replace('@', '') for line in usernameFile
                                                  if line.strip()]
      usernameFile.close()
      
      bunches = _mkBunches(users)
      
      todo = True
      while todo or options.iter:
        todo = False
        for bunch in bunches:
          infos = getUserData(bunch)
          
          outFile = gzip.open(os.path.join(OUT_DIR,
                         'userInfo_%s_to_%s_%s.pickle.gz' %
                         (bunch[0], bunch[-1], getTimeString())), 'wb')
          pickle.dump([i for u, i in infos], outFile)
          outFile.close()
        if options.iter:
          zzz(ITER_SNOOZE)
    elif options.search_kws:
      if not options.sinceId:
        print('Must input "since status ID" when searching.')
      else:
        kwSearch(KW_PATH, int(options.sinceId))
    elif options.retrieve_user_timeline:
      statusCollecter = Collect()
      statusCollecter.getPastTweets()
    elif options.retrieve_retweets:
      getRTs(KW_PATH)
    elif options.retrieve_statuses:
      getStatuses(KW_PATH)
    elif options.geo_search:
      getGeoLocations(KW_PATH)
    else:
      print('Need to at least pass -k, -u, -g, -p, -i, -s, -r, or -n.\n \
             Call with --help for list of options')
  except Exception as ex:
      try:
        tb = str(traceback.format_exc())
      except:
        tb = ''
      emailAlert('%s\n%s\n%s\n\n' % (str(sys.exc_info()), tb, str(ex)))
  
