import pandas as pd

from VegaZero2VegaLite import VegaZero2VegaLite

vz2vl = VegaZero2VegaLite()


def validate_vega_lite(vega_zero):
    try:
        vz2vl.to_VegaLite(vega_zero)
        return True
    except:
        return False


def remove_wrong_vega_zero(df):
    df = df[df['labels'].apply(validate_vega_lite)].reset_index(drop=True)

    return df


def filter_vega_zero():
    for file in ['dev', 'train', 'test']:
        df = pd.read_csv(f'./dataset/dataset_final/{file}.csv')
        df.to_csv(f'./dataset/dataset_final/old_{file}.csv', index=False)
        df = remove_wrong_vega_zero(df)
        df.to_csv(f'./dataset/dataset_final/{file}.csv', index=False)


if __name__ == '__main__':
    filter_vega_zero()
