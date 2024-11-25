import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_args, read_yaml

tqdm.pandas()

def filter_success_download_file(df, path):
    file_exist_bool_list = []
    for file_path in tqdm(df['file_path'].values):
        file_exist_bool = os.path.exists(f'{path}/{file_path}')
        file_exist_bool_list.append(file_exist_bool)
    print(f'[Before] Filter Success Download Files : {df.shape[0]}')
    df['is_exists'] = file_exist_bool_list
    df = df[df['is_exists']]
    df.drop(columns=['is_exists'], axis=1, inplace=True)
    print(f'[After] Filter Success Download Files : {df.shape[0]}')
    return df


def filter_one_to_one_matching_by_collection_idx(df, threshold):
    print(f'[Before] Filter One-to-one Collection Files : {df.shape[0]}')
    value_counts = df['element_idx'].value_counts()
    filtered_values = value_counts[value_counts >= threshold].index
    df = df[~df['element_idx'].isin(filtered_values)]
    print(f'[After] Filter One-to-one Collection Files : {df.shape[0]}')
    return df


def filter_collection_idx_by_count(df, threshold):
    print(f'[Before] Filter Collection Count : {df.shape[0]}')
    value_counts = df['collection_idx'].value_counts()
    filtered_values = value_counts[value_counts >= threshold].index
    df = df[~df['collection_idx'].isin(filtered_values)]
    print(f'[After] Filter Collection Count : {df.shape[0]}')
    return df


def main():
    args = get_args()
    config = read_yaml(args.config_path)
    elements_df = pd.read_csv(f'{config["CSV_PATH"]}/{config["COLLECTION_FILE_NAME"]}')
    designer_keywords_df = pd.read_csv(f'{config["CSV_PATH"]}/{config["DESIGNER_FILE_NAME"]}')

    elements_df.dropna(inplace=True)
    elements_df['element_idx'] = elements_df['element_idx'].progress_apply(lambda x: str(int(x)))
    elements_df['file_path'] = elements_df['file_path'].progress_apply(lambda x: f'{x}.png')

    elements_df = filter_success_download_file(elements_df, config['IMAGE_SAVE_PATH'])
    elements_df = filter_one_to_one_matching_by_collection_idx(elements_df, config['THRESHOLD_COLLECTION_PER_ELEMENT'])

    designer_keywords_df.dropna(inplace=True)
    designer_keywords_df['element_idx'] = designer_keywords_df['element_idx'].progress_apply(lambda x: str(int(x)))
    designer_keywords_df['keywords'] = designer_keywords_df['keywords'].progress_apply(
        lambda x: list(set(x.split('|'))))
    designer_keywords_df['file_path'] = designer_keywords_df['file_path'].progress_apply(lambda x: f'{x}.png')

    elements_df = pd.merge(elements_df, designer_keywords_df, on=['collection_idx', 'element_idx', 'element_type',
                                                                  'file_path'], how='left')

    elements_df.fillna(f'{config["NOT_EXISTS_SYMBOL"]}', inplace=True)
    collection_idx_list = elements_df['collection_idx'].unique()

    train_collection_idx, test_collection_idx = train_test_split(collection_idx_list,
                                                                 test_size=f'{config["TEST_RATIO"]}',
                                                                 random_state=f'{config["SEED"]}')

    train_elements_df, test_elements_df = elements_df[elements_df.collection_idx.isin(train_collection_idx)], \
        elements_df[elements_df.collection_idx.isin(test_collection_idx)]
    print(f'Train Collection : {train_collection_idx.shape[0]} | Test Collection : {test_collection_idx.shape[0]}')
    print(f'Train Elements : {train_elements_df.shape[0]} | Test Elements : {test_elements_df.shape[0]}')

    elements_df = filter_collection_idx_by_count(elements_df, config['THRESHOLD_COLLECTION_COUNT'])
    train_elements_df = filter_collection_idx_by_count(train_elements_df, config['THRESHOLD_COLLECTION_COUNT'])
    test_elements_df = filter_collection_idx_by_count(test_elements_df, config['THRESHOLD_COLLECTION_COUNT'])

    elements_df.to_csv(f'{config["CSV_PATH"]}/total_dataset.csv', index=False)
    train_elements_df.to_csv(f'{config["CSV_PATH"]}/train_dataset.csv', index=False)
    test_elements_df.to_csv(f'{config["CSV_PATH"]}/test_dataset.csv', index=False)


if __name__ == '__main__':
    main()
