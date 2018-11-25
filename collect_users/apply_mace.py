'''
Use MACE to reconcile user labels.  Also computes agreement with gold labels
(Annabelle labeled majority category for subset of users).

Adrian Benton
11/11/2018
'''

import argparse
import numpy as np
import os
import pandas as pd
import re
import sklearn
import sklearn.metrics

from collect_users.krippendorff_alpha import krippendorff_alpha, nominal_metric

# AB: Point to MACE executable script.  This points to it on my local machine.
MACE = '/home/annabelle/PycharmProjects/MACE/MACE'

MACE_INPUT_PATH_RE = re.compile('mace_input(?P<suffix>_\w+\-Answer\.(?P<label>\w+)\-full_rebinned\.csv)')

# temporary files 
TMP_PREFIX = './tmp'

def main(input_dir, gold_path, output_prefix):
    ps = [os.path.join(input_dir, p) for p in os.listdir(input_dir)
          if MACE_INPUT_PATH_RE.match(os.path.basename(p))]
    
    df = pd.read_table(gold_path, sep=',')
    gold_uid_to_category = {r['Input.user_id']: r['Answer.category_most_index']
                            for _, r in df.iterrows()
    }
    gold_uid_to_clazz = {r['Input.user_id']: r['Answer.classify_account']
                            for _, r in df.iterrows()
    }
    
    label_path = output_prefix + '.labels.tsv'
    desc_stat_path = output_prefix + '.desc_stats.txt'
    pdf_path = output_prefix + '.plots.pdf'
    
    user_to_labels = {}
    worker_agreement = {}

    desc_stat_file = open(desc_stat_path, 'wt', encoding='utf8')
    
    for p in ps:

        # run MACE and get distribution over labels
        cmd = ('{} --restarts 10 --distribution --entropies --prefix {} {}'.
               format(MACE,
                      TMP_PREFIX,
                      p
               )
        )
        print(cmd)
        os.system(cmd)
        
        m = MACE_INPUT_PATH_RE.match(os.path.basename(p))
        
        # read in list of user IDs
        uid_path = os.path.join(os.path.dirname(p),
                                 'userids' + m.group('suffix'))
        
        users = pd.read_table(uid_path)['user_ids']
        
        for u in users:
            if u not in user_to_labels:
                user_to_labels[u] = {'user_id': u}
        
        label = m.group('label')

        # add resolved label to each user
        f = open(TMP_PREFIX + '.prediction', 'rt')
        for u, ln in zip(users, f):
            flds = [(' '.join(v_fld.split()[:-1]), float(v_fld.split()[-1]))
                    for v_fld in ln.strip().split('\t')]

            # if there is no clear winner, leave this field unlabeled
            if flds[0][1] <= flds[1][1]:
                user_to_labels[u]['{}-mace_label'.format(label)] = None
            else:
                user_to_labels[u]['{}-mace_label'.format(label)] = flds[0][0]
            user_to_labels[u][
                '{}-mace_label_distribution'.format(label)
            ] = dict(flds)
        f.close()
        
        # compute Krippendorff's alpha agreement for this label
        all_annotations = pd.read_table(p, header=None, sep=',')
        all_annotations = np.array(all_annotations).T.tolist()
        all_annotations = [[v if type(v) == str else None for v in vs]
                           for vs in all_annotations]
        krip_alpha = krippendorff_alpha(all_annotations,
                                        metric=nominal_metric,
                                        force_vecmath=False,
                                        convert_items=lambda x: x,
                                        missing_items={None})
        
        desc_stat_file.write('='*10 +
                             ' Stats for "{}" '.format(label) +
                             '='*10 + '\n')
        desc_stat_file.write('Krippendorff\'s alpha: {}\n'.format(
            krip_alpha)
        )
        
        label_dist = {}
        for label_map in user_to_labels.values():
            mace_label = label_map['{}-mace_label'.format(label)]
            if mace_label not in label_dist:
                label_dist[mace_label] = 0
            label_dist[mace_label] += 1
        desc_stat_file.write(
            'Label distribution: {} {}\n'.format(label_dist,
                                    {k: v/sum(label_dist.values())
                                     for k, v in label_dist.items()})
        )
        
        # we have gold labels only for the main specialization
        # category for 200-odd users, compute agreement for them
        num_agree = 0
        denom = 0
        num_unsure = 0
        if label in {'category_most_index', 'classify_account'}:
            gold_uid_to_label = gold_uid_to_category if label == 'category_most_index' else gold_uid_to_clazz
            
            gold_labels = [gold_uid_to_label[u].strip().lower()
                           for u in sorted(list(gold_uid_to_label.keys()))
                           if gold_uid_to_label[u] is not None and
                           user_to_labels[u]['{}-mace_label'.format(label)] is not None]
            mace_labels = [user_to_labels[u]['{}-mace_label'.format(label)].strip().lower()
                           for u in sorted(list(gold_uid_to_label.keys()))
                           if gold_uid_to_label[u] is not None and
                           user_to_labels[u]['{}-mace_label'.format(label)] is not None]
            
            label_names = sorted(list(set(gold_labels + mace_labels)))
            conf_max = sklearn.metrics.confusion_matrix(gold_labels, mace_labels,
                                                        labels=label_names)
            conf_max_df = pd.DataFrame(data=conf_max, index=label_names, columns=label_names)
            
            for u, gold_label in gold_uid_to_label.items():
                mace_label = user_to_labels[u]['{}-mace_label'.format(label)]
                
                if mace_label is None:
                    num_unsure += 1
                else:
                    denom += 1
                    
                    if mace_label == gold_label.strip().lower():
                        num_agree += 1
    
            desc_stat_file.write('='*10 +
                                 ' Gold Agreement for "{}" '.format(label) +
                                 '='*10 + '\n')
            desc_stat_file.write(
                '# MACE uncertain: {}/{}\n'.format(num_unsure,
                                                   len(gold_uid_to_label))
            )
            desc_stat_file.write(
                '# Agree: {}/{} ({:.2f}%)\n'.format(num_agree,
                                                    denom,
                                                    100 * num_agree / float(denom))
            )
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                desc_stat_file.write('Confusion matrix:\n{}\n'.format(conf_max_df))
    
    desc_stat_file.close()

    df = pd.DataFrame(list(user_to_labels.values()))
    df.to_csv(label_path, sep='\t', header=True, index=False)

if __name__ == '__main__':
    """
    Generates MACE input file from HIT result file
    """
    
    parser = argparse.ArgumentParser(
      description='apply MACE to reconcile annotator labels'
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with MACE-formatted input files')
    parser.add_argument('--gold_path', required=True,
                        dest='gold_path', metavar='GOLD_PATH',
                        help='path to user gold labels')
    parser.add_argument('--output_prefix', required=True,
                        dest='output_prefix', metavar='OUTPUT_PREFIX',
                        help='prefix to write out final labels, ' +
                             'descriptive statistics, and plots')
    args = parser.parse_args()
    
    in_dir = args.input_dir
    gold_path = args.gold_path
    out_prefix = args.output_prefix
    
    main(in_dir, gold_path, out_prefix)

