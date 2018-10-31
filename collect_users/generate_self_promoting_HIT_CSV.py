from configs.config import *
from utils.sample_util import *
from utils.file_io import *

import os

inactive_id = '1YjSjnArcKlmtIyNbVUimjRhnXeGwXHpL'
not_english_id = '1qgZvxRAbVk0iBCJBB73I4CrwxbKbA_zd'
personal_id = '1iH5Tv1OmovyhiwuvRWiOVioIWlGBhQlS'
brand_id = '1u3n3mMmYcFCTMD5Yuhax2Xr1bYcGQU5r'
travel_id = '11gtzS6xvMctmzjDfkAy6jKH6ERjnt1hx'
beauty_id = '16jQKjFZoCrU0GjAv-L3lnn3zDVV76etg'
gastro_id = '1eD8jx_qa7IF4_4bcmTkNgdrveLajXq89'
politics_id = '1rJQsnbwTtoXFTRq37ATFSTGaJaPn2uX1'
style_id = '13lIg99ukplqrelqPpYNnqvtLj38ymZgp'
family_id = '1fKsoDk_FvW0kAoExhq_d7FelMt7xGXvU'
sports_id = '1Ea005AV1cIe8fceJEyTCcYxNar59TUuJ'
other_id = '1gS6sArS_N21Cv52gwCdutF8dIq0Ky0gy'
health_id = '1wzkO_7V9ZEGkPeumcZtbwE0zbtFp03cg'
arts_id = '1t5g7ATcuKVXgXwztO0df_ELwrilNNGi-'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    parser.add_argument("num_hits", help="number of hits")
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    sample_size = int(args.num_hits)

    user_ids_shuffled = users_from_sample(in_dir, sample_size)

    csv_rows = []
    column_headers = ['inactive_id', 'not_english_id', 'personal_id', 'brand_id', 'travel_id', 'beauty_id', 'gastro_id', 'politics_id', 'style_id', 'family_id', 'sports_id', 'health_id', 'arts_id', 'other_id', 'user_id']
    csv_rows.append(column_headers)

    for id in user_ids_shuffled:
        row = [inactive_id, not_english_id, personal_id, brand_id, travel_id, beauty_id, gastro_id, politics_id, style_id, family_id, sports_id, health_id, arts_id, other_id, str(id)]
        csv_rows.append(row)

    csv_fname = os.path.join(out_dir, str(sample_size) + '_size_HIT_self_promoting.csv')
    write_rows_to_csv(csv_fname, csv_rows)
