'''
Repair MTurk annotations.  Assumes that all MTurk results batches are joined in a single file
(full.csv).

Adrian Benton
11/9/2018
'''

import argparse
import numpy as np
import pandas as pd


# These mappings from category to text descriptions were generated manually.
# They seemed right, and new categories that capture very few users are discarded.  --Adrian
new_category_mapping = {
    'lifestyle': {'lifestyle', 'lifestyle blogger', 'lifetyle'},
    'science and technology': {'cyber security', 'technology', 'tech', 'information technology',
                               'computer', 'computers', 'computer science',
                               'computer engineering blog', 'technology/developer',
                               'technology, science', 'technology, internet',
                               'technology programming', 'technical', 'software engineer',
                               'web development', 'web developer', 'tech blogging',
                               'engineering', 'mathematics', 'electronics', 'developing skills',
                               'it', 'cyber security', 'science'},
    'business': {'marketing', 'business', 'digital marketing', 'entrepreneur',
                 'entrepreneurship', 'internet marketing', 'online marketing',
                 'business,consultant', 'business, social network',
                 'business, motivational speaker', 'business, blogger'},
    'books': {'books', 'book blogger', 'literature', 'books/reading', 'book reviews'},
    'religion': {'religion', 'faith', 'spiritual', 'spirituality', 'religious',
                 'religion, religious books', 'religion (Christianity)'},
    'games': {'videogames and streaming', 'videogames', 'video_blogger', 'video games',
              'gambling', 'game', 'gaming', 'affiliate twitch streamer', 'gamer', 'games',
              'gaming, streaming', 'gambling', 'video games', 'blogger, gamer', 'gaming'},
    'finance': {'finance', 'stock trading', 'stock market', 'cryptocurrency',
                'crypto currency' 'taxes', 'finance blogger', 'investments',
                'financial/ stock market', 'finance (cryptocurrency)', 'finance (taxes)',
                'finance blogger', 'money', 'personal finance', 'making money', 'finances'}
}

old_category_mapping = {
    'arts': {"graphic designer, blogger", 'graphic design', 'artist',
             'photography', 'photographer', 'photographer, blogger', 'music',
             'blogger, music', 'singing/music', 'poetry', 'poet, writer, blogger',
             'singer', 'poetry, writing', 'song writer', 'photos', 'photogragher',
             'music (rap)'},
    'style': {'fashion', 'designer', 'blogger, designer', 'blogger, fashion'},
    'family': {'parenting', 'homeschooling'},
    'health': {'mental health', 'medical', 'mental health and abuse'},
    'gastronomy': {'food', 'cooking'},
    'politics': {'social activism', 'socialist', 'criminal justice', 'activist'},
    'sports': {'sports radio host'}
}

txt_to_category = dict([(v, k) for k, vs in new_category_mapping.items() for v in vs] +
                       [(v, k) for k, vs in old_category_mapping.items() for v in vs])

all_old_categories = ['arts', 'beauty', 'family', 'gastronomy', 'health', 'politics', 'sports', 'style', 'travel']
all_categories = all_old_categories + list(new_category_mapping.keys())

EMPTY = np.float('nan')

def main(args):
    REBINNED = 'rebinned_category_most_index'
    ORIG_MAIN = 'Answer.category_most_index'
    
    df = pd.read_table(args.inpath, dtype=object, sep=',')

    def _row_to_rebinned_category(row):
        most_cat = row[ORIG_MAIN]
        
        # If annotator selected a category, trust them
        if most_cat != 'other':
            return most_cat
        else:
            other_text = row['Answer.other_text'].lower().strip()
            if other_text in txt_to_category:
                return txt_to_category[other_text]
            else:
                return 'other'
    
    # (1) Assign category labels for Other specializations that match mappings.
    df[REBINNED] = [_row_to_rebinned_category(row)
                                          for _, row
                                          in df.iterrows()]
    num_main_remapped = (df[REBINNED]!=
                         df[ORIG_MAIN]).sum()
    # (1.1) Add binary indicators for all_* for new categories
    for c in new_category_mapping:
        df['Answer.category_all_{}'.format(c)] = ['category_all_{}'.format(c)
                                                  if row[REBINNED]==c
                                                    else EMPTY
                                                  for _, row in df.iterrows()]
    for c in old_category_mapping:
        df['Answer.category_all_{}'.format(c)] = ['category_all_{}'.format(c)
                                                  if row[REBINNED]==c
                                                    else EMPTY
                                                  for _, row in df.iterrows()]
    
    #### (2) If tagged with "other_all" and another category, drop "other_all
    ###new_other_all = [None
    ###                 if np.isnan(row['Answer.category_all_other'])
    ###                 else (None if any([np.isnan(row['Answer.category_all_{}'.format(c)])
    ###                                    for c in all_categories]) else 'category_other')
    ###                 for i, row in df.iterrows()]
    ###df['Answer.category_all_other'] = new_other_all
    
    # (2) If only one "*_all" tagged and no main specialization, copy this to
    #     main specialization.

    num_missing = 0
    num_missing_multi = 0
    main_spec = df[REBINNED]
    for i, row in df.iterrows():
        
        if (row['Answer.classify_account'] == 'promoting') and \
           (type(row['Answer.category_most_index'])!=str):
            selected_categories = [c for c in all_categories
                                   if type(row['Answer.category_all_{}'.format(c)])==str]
            if len(selected_categories) == 1:
                main_spec[i] = c
            elif len(selected_categories) > 1:
                print('Multiple categories for annotation # {}: {}'.format(i,
                                                                           selected_categories))
                num_missing_multi += 1
            else:
                print('Cannot find a good category for annotation # {}'.format(i))
                num_missing += 1

    print('Missing: 0: {}, >1: {}, Remapped main spec: {}'.format(
        num_missing, num_missing_multi, num_main_remapped)
    )
    df[REBINNED] = main_spec
    
    df[ORIG_MAIN] = df[REBINNED]
    del df[REBINNED]

    import pdb; pdb.set_trace()
    
    df.to_csv(args.outpath, sep=',', index=False, header=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='repair MTurk results')
    parser.add_argument('--inpath', dest='inpath', default='../full.csv',
                        help='points to the input path', metavar='INPUT_PATH')
    parser.add_argument('--outpath', dest='outpath', default='full_rebinned.csv',
                        help='points to the output path', metavar='OUTPUT_PATH')

    args = parser.parse_args()
    main(args)
    
