import re
import requests

#user json fields, by type
ints = ['followers_count', 'friends_count', 'listed_count', 'retweet_count', 'favorite_count']
objs = ['user','status']

def has_linked_page(user):
    url_re = re.compile('.*http.|.*www.')
    description = field_from_json('description', user)
    url = field_from_json('url', user)

    if url:
        full_url = url_re.findall(url)
        if full_url:
            return True
    if description:
        url_in_description = url_re.findall(description)
        if url_in_description:
            return True
    return False

def is_active_id(id):
    response = requests.get('https://twitter.com/intent/user?user_id=' + id)
    if response.status_code == 200:
        return True
    return False


def is_active(user):
    id = field_from_json('id_str', user)
    active_user = is_active_id(id)

    return active_user

def re_for_json_field(field):
    if field in ints:
        return re.compile(r'"{}":.*?(\d+)'.format(field), re.S)
    elif field == 'user':
        return re.compile(r'"user":.*?(\{.+?\})', re.S)
    elif field == 'status':
        return re.compile(r'"status":.*?(\{.+?\})', re.S)
    else:
        return re.compile(r'"{}":.*?"(.+?)"'.format(field), re.S)

def field_from_json(field, json):
    field_in_json = re_for_json_field(field).findall(json)
    if field_in_json:
        return field_in_json[0]
    else:
        return None

def fields_for_mturk(json):
    mturk_fields = ['id_str', 'name', 'screen_name', 'description', 'followers_count']
    return [field_from_json(field, json) for field in mturk_fields]