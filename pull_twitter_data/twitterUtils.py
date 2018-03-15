'''
Mapping from common Twitter error codes to exceptions.

Adrian Benton
4/15/2013
'''

class TweepError(Exception):
  def __init__(self, reason, response):
    Exception.__init__(self, '%d: %s' % (response, reason))

class NotModTweepError(TweepError):
	'''
	Corresponds to error code 304 Not Modified: There was no new data to return.
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

class RateLimitTweepError(TweepError):
	'''
	Corresponds to error codes 400, 420.  Bad Request and Enhance Your Calm
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

class UnauthTweepError(TweepError):
	'''
	Corresponds to error code 401 Unauthorized: Authentication credentials were missing or incorrect.
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

class ForbiddenTweepError(TweepError):
	'''
	Corresponds to error code 403 Forbidden: Used when requests are denied
	due to update limits.
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

class NotFoundTweepError(TweepError):
	'''
	Corresponds to error code 404 Not found: Used when user is non-existent.
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

class NotAcceptableTweepError(TweepError):
	'''
	Corresponds to error code 406 Not Acceptable: Returned by the Search API
	when an invalid format is specified in the request.
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

class ServerTweepError(TweepError):
	'''
	Corresponds to error codes 500, 502, and 503.  Internal Server Error,
	Bad Gateway, and Service Unavailable.
	'''
	def __init__(self, reason, response):
		TweepError.__init__(self, reason, response)

'''
Mapping from response codes to less a more fine-grained TweepError.
'''
errorMap = {304:NotModTweepError, 400:RateLimitTweepError,
	    401:UnauthTweepError, 403:ForbiddenTweepError,
	    404:NotFoundTweepError, 406:NotAcceptableTweepError,
	    420:RateLimitTweepError, 500:ServerTweepError,
	    502:ServerTweepError, 503:ServerTweepError}

def getRefinedTweepError(e):
	'''
	Returns a more specific TweepError to assist error handling.
	If no HTTP error was thrown, then just return the Tweep error.
	'''
	if e.response and e.response.status in errorMap:
		return errorMap[e.response.status](e.reason, e.response.status)
	else:
		return e
