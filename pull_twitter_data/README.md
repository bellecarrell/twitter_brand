# Twitter collection tool #

## Dependencies ##

- [Python 2.7](http://www.python.org/download/releases/2.7/)
- [tweepy](https://github.com/tweepy/tweepy), included locally

## Scripts ##

**twitter_search.py**

### Function ###

Swiss army knife of Twitter data collection.  This script can hit a variety of
methods of the Twitter REST and Streaming APIs.  Below are what it can collect:

- Retrieve user IDs given screennames.  The output directory will contain a
  tab-separated file containing the usernames and user IDs of the people to
  collect IDs for.
- Retrieve all user information for a set of usernames/user IDs.  This can be
  set to collect repeatedly, or just perform a single pass.  The output files
  consist of pickled lists of *Tweepy User* objects, batched at a maximum of
  one hundred per output file.
- Retrieve all retweets made in response to a set of status IDs.  Each output
  file corresponds to a single status, and contains a pickled list of 
  *Tweepy Status* objects.  For most statuses, these will be empty lists
- Sample network: Samples/collects the local network for a list of
  IDs/usernames (both followers and friends).  This can be set to repeat
  indefinitely or just a single pass.  The output are dumped to separate files
  for followers and friends where each line is of the form:
    > *TIME\_COLLECTED    SOURCE\_USER    NEIGHBOR\_1    ...    NEIGHBOR\_N*
- Given a list of usernames/user IDs, collect the most recent past tweets these
  users made (currently the API lets you look back up to 3200 tweets).  Each
  output file corresponds to a particular user and contains a pickled list of
  *Tweepy Status* objects, that user's most recent tweets.
- Given a set of keywords, will attempt to collect tweets mentioning these
  keywords.  This search is constrained by the *since_id* parameter, which
  determined what the oldest ID you want to fetch is.  The tweets and number
  of tweets returned are limited by the Twitter API.  The output is lists of
  pickled *Tweepy SearchResult* objects.  These are similar to *Status*
  objects, but with a reduced set of fields.  The name of each file corresponds
  to each of the keywords searched for.
- Given a list of user IDs, tap into the user stream for these users.  This
  will stream in tweets that are by or mention these users.  The output
  consists of pickled lists of *Tweepy Status* objects, batched by a maximum
  of *NUM_TO_CACHE* tweets per file.
- Given a set of keywords, will stream tweets that mention any of these
  keywords.  These tweets can also be constrained by geolocation by setting
  the *location* parameter of the script.  The output consists of pickled lists
  of *Tweepy Status* objects, batched by a maximum of *NUM_TO_CACHE* tweets per
  file.

### Arguments ###


#### Obligatory arguments ####

These arguments are required by the script, regardless of the method called.

- *--keypath*: Path to where your consumer key and secret are stored.  The file
               should contain your consumer key on the first line of the file,
			   and consumer secret on the second line. **obligatory**
- *--accesstoken*: Path to where your access token and access token secret are
                   stored.  Should contain the token on the first line, and
				   secret on the second. **obligatory**
- *--kwpath*: Path to where the input list of status IDs/user IDs/usernames/
              keywords you want to search for are stored.  Just a list of items
			  with one item per line.  Parsed appropriately given the method
			  called. **obligatory**
- *--outdir*: Path to the directory where your data should be written to.  This
              directory need not exist yet, will be created by script.
			  **obligatory**

#### Method-specific parameters ####

These parameters may be set depending on the method.

- *--sinceid*: Set for *keyword search* to determine the earliest tweet that
               should be returned.  This is a status ID (long integer).
			   **obligatory**
- *--location*: Set for the *user stream* and *keyword stream* methods.  This
                will restrict the tweets that you collect to only those
				occurring in a list of bounding boxes supplied on the command
				line.  Each bounding box is specified by four comma-delimited
				floats, and the bounding boxes are separated by commas.  For
				each bounding box, specify first the latitude/longitude of the
				southwest corner, then the latitude/longitude of the northeast
				corner.
- *--numtocache*: Set for the *user stream*, *keyword stream*, and
                  *user information* methods.  For the streaming methods, this
				  determines the maximum number of tweets that are stored per
				  file.  For the *user information* collection, this
				  determines the maximum number of tweets to collect for each
				  user (up to 3200).  Value is an integer.  Defaults to *100*.
- *--iter*: For the *user information* and *sample network* methods, setting
            this flag will cause the script to sample information for the input
			users repeatedly.  Otherwise, these methods will just make a single
			pass over the input users.  Used when tracking network change
			over time, for instance.

#### Method flags ####

One of these must be set to determine the operation you want to perform over
the Twitter APIs.

- *--kwstream*: Stream in tweets containing any of a list of keywords.
- *--userfollow*: Follow the user streams for a set of users, collecting tweets
                  regarding them.
- *--netsample*: Sample the local network of a collection of users (followers/
                 friends).
- *--getids*: For a collection of usernames, dumps those users' IDs.  Used
              prior to *--userfollow*, for instance.
- *--info*: Collects user information for a set of usernames/user IDs.
- *--search*: Perform a keyword search, grabbing most recent tweets made
              containing any of a set of keywords.
- *--pasttweets*: Collect the most recent tweets that a set of users have made.
- *--retweets*: Collect all retweets that have been made in response to a set
                of status IDs.

### Sample usage ###

<!-- Sample call for keyword stream, restricted to New York City -->

	  python twitter_search.py --kwstream --keypath=keys/ac_keys.txt --accesstoken=access_tokens/access_token_ac.txt --kwpath=lists/test_kws.txt --outdir=keyword_stream --numtocache=10 --location=40.513799,-74.311523,41.153842,-71.850586

<!-- Sample call for user stream -->

	  python twitter_search.py --userfollow --keypath=keys/cr2_keys.txt --accesstoken=access_tokens/access_token_cr2.txt --kwpath=lists/test_user_ids.txt --outdir=user_stream --numtocache=1

<!-- Sample call for network collection, will sample indefinitely -->

	  python twitter_search.py --netsample --keypath=keys/dl1_keys.txt --accesstoken=access_tokens/access_token_dl1.txt --kwpath=lists/test_handles.txt --outdir=networks --iter

<!-- Sample call for retrieving user IDs -->

	  python twitter_search.py --getids --keypath=keys/dl2_keys.txt --accesstoken=access_tokens/access_token_dl2.txt --kwpath=lists/test_handles.txt --outdir=user_ids

<!-- Sample call for collecting user information, only performs single pass -->

      python twitter_search.py --info --keypath=keys/dl3_keys.txt --accesstoken=access_tokens/access_token_dl3.txt --kwpath=lists/test_handles.txt --outdir=user_infos

<!-- Sample call for keyword search, stopping at the ID in "sinceid" --> 

      python twitter_search.py --search --keypath=keys/dl4_keys.txt --accesstoken=access_tokens/access_token_dl4.txt --kwpath=lists/test_kws.txt --outdir=keyword_search --sinceid=312378025792655360

<!-- Sample call for collecting users' most recent tweets, up to 1000 -->

      python twitter_search.py --pasttweets --keypath=keys/t1_keys.txt --accesstoken=access_tokens/access_token_t1.txt --kwpath=lists/test_handles.txt --outdir=user_past_tweets --numtocache=1000

<!-- Sample call for collecting the retweets resulting from a set of IDs -->

      python twitter_search.py --retweets --keypath=keys/t2_keys.txt --accesstoken=access_tokens/access_token_t2.txt --kwpath=lists/test_status_ids.txt --outdir=status_retweets
