import pandas as pd
from tqdm import tqdm
from utils import get_args, read_yaml


def main():
    args = get_args()
    config = read_yaml(args.config_path)
    elements_df = pd.read_csv(f'{config["CSV_INPUT_PATH"]}/{config["ELEMENTS_FILE_NAME"]}')
    test_elements_df = pd.read_csv(f'{config["CSV_INPUT_PATH"]}/{config["TEST_ELEMENTS_FILE_NAME"]}')
    flatten_elements_df = elements_df.explode('keywords')
    flatten_test_elements_df = test_elements_df.explode('keywords')

    keywords_values = flatten_elements_df[
        flatten_elements_df.keywords != f'{config["NOT_EXISTS_SYMBOL"]}'].keywords.value_counts()
    filtered_keywords_values = keywords_values[(keywords_values >= config["MIN_ELEMENT_NUM"]) &
                                               (keywords_values <= config["MAX_ELEMENT_NUM"])]

    search_test_elements_df = flatten_test_elements_df[
        flatten_test_elements_df.keywords.isin(filtered_keywords_values.keys())]
    print(f'[Before] Filter Frequency Use Keyword : {flatten_elements_df.shape[0]}')
    flatten_elements_df = flatten_elements_df[flatten_elements_df.keywords.isin(filtered_keywords_values.keys())]
    print(f'[After] Filter Frequency Use Keyword : {flatten_elements_df.shape[0]}')

    search_test_dataset = []
    num_of_samples = 9999999
    for i, search_word in tqdm(enumerate(filtered_keywords_values.keys()), total=len(filtered_keywords_values)):
        query_df = search_test_elements_df[search_test_elements_df.keywords == search_word]
        query_df = query_df.sample(n=min([num_of_samples, query_df.shape[0]]), random_state=i)
        for q_collection_idx, q_element_idx, q_element_type, q_file_path, _ in query_df.values:
            case_flatten_elements_df = flatten_elements_df[
                (flatten_elements_df.file_path != q_file_path) & (flatten_elements_df.keywords == search_word)]
            answer_case_flatten_elements_df = case_flatten_elements_df[
                case_flatten_elements_df.collection_idx == q_collection_idx]
            search_test_dataset.append([search_word, q_collection_idx, q_element_idx, q_element_type, q_file_path,
                                        case_flatten_elements_df['collection_idx'].values,
                                        case_flatten_elements_df['element_idx'].values,
                                        case_flatten_elements_df['element_type'].values,
                                        case_flatten_elements_df['file_path'].values,
                                        answer_case_flatten_elements_df['collection_idx'].values,
                                        answer_case_flatten_elements_df['element_idx'].values,
                                        answer_case_flatten_elements_df['element_type'].values,
                                        answer_case_flatten_elements_df['file_path'].values])
    search_test_dataset_df = pd.DataFrame(search_test_dataset,
                                          columns=['search_word', 'q_collection_idx', 'q_element_idx', 'q_element_type',
                                                   'q_file_path',
                                                   'c_collection_idx', 'c_element_idx', 'c_element_type', 'c_file_path',
                                                   'a_collection_idx', 'a_element_idx', 'a_element_type',
                                                   'a_file_path'])
    print(f'Original Search Test Dataset : {search_test_dataset_df.shape[0]}')
    search_test_dataset_df.to_csv(f'{config["CSV_INPUT_PATH"]}/all_search_test_dataset.csv')

    print(f'Filter Search Test Dataset : {search_test_dataset_df.shape[0]}')
