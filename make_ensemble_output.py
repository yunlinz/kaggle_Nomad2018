import pandas as pd
import numpy as np
import  os

def assemble(outdir):
    df = None
    for dir in os.listdir(outdir):
        if dir in ('8','7','6', '4', '1'):
            new_df = pd.read_csv(outdir + '/' + dir + '/submission.csv')
            if df is None:
                df = new_df
            else:
                df = df.append(new_df, ignore_index=True)
    print('averaging {} entries'.format(len(df)))
    df.groupby('id').mean().to_csv(outdir + '/submission_cherrypicked.csv')


def diagnose_validation(outdir):
    df = None
    for dir in os.listdir(outdir):
        if os.path.isdir(outdir + '/' + dir):
            new_df = pd.read_csv(outdir + '/' + dir + '/valid_diag.csv')
            if df is None:
                df = new_df
            else:
                df = df.append(new_df, ignore_index=True)

    df.to_csv(outdir + '/valid_diag.csv')

if __name__ == '__main__':
    diagnose_validation('output/v0.8')