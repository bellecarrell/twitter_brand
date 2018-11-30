import datetime

#https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates
def randomDate(start, end, prop):
    stime, etime = start, end
    ptime = stime + prop * (etime - stime)
    return ptime

def days_between(d1, d2):
    return abs((d2 - d1).days)

def time_window(date,window_size):
    tw_delta = datetime.timedelta(days=window_size)
    stop = date + tw_delta
    return (date, stop)

def posted_recently(collection_date,user_date):
    return datetime.datetime.fromtimestamp(collection_date) - datetime.timedelta(days=60) <= datetime.datetime.fromtimestamp(user_date)