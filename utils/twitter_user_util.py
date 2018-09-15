import re

#user json fields, by type
ints = ['followers_count']

def re_for_json_field(field):
    if field in ints:
        return re.compile(r'"{}":.*?(\d+)'.format(field), re.S)
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