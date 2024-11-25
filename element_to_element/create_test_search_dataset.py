import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_args, read_yaml

tqdm.pandas()


def main():
    args = get_args()
    config = read_yaml(args.config_path)
    elements_df = pd.read_csv(f'{config["CSV_INPUT_PATH"]}/{config["ELEMENTS_FILE_NAME"]}')
    test_elements_df = pd.read_csv(f'{config["CSV_INPUT_PATH"]}/{config["TEST_ELEMENTS_FILE_NAME"]}')
    elements_df = elements_df[elements_df.keywords != config['NOT_EXISTS_SYMBOL']]
    test_elements_df = test_elements_df[test_elements_df.keywords != config['NOT_EXISTS_SYMBOL']]
    elements_df['keywords'] = elements_df['keywords'].progress_apply(lambda x: eval(x))
    test_elements_df['keywords'] = test_elements_df['keywords'].progress_apply(lambda x: eval(x))
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
    for i, search_word in tqdm(enumerate(filtered_keywords_values.keys()), total=len(filtered_keywords_values)):
        query_df = search_test_elements_df[search_test_elements_df.keywords == search_word]
        query_df = query_df.sample(n=min([config["NUM_OF_SAMPLES"], query_df.shape[0]]), random_state=i)
        for q_collection_idx, q_element_idx, q_element_type, q_file_path, _, _ in query_df.values:
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

    search_test_dataset_df = search_test_dataset_df[search_test_dataset_df.a_collection_idx.str.len() != 0]
    print(f'Filter Not Exists Answer Search Test Dataset : {search_test_dataset_df.shape[0]}')
    search_test_dataset_df.to_csv(f'{config["CSV_INPUT_PATH"]}/search_test_dataset.csv')

    search_test_dataset_df['candidate_answer_ratio'] = search_test_dataset_df.progress_apply(
        lambda x: len(x['a_collection_idx']) / len(x['c_collection_idx']), axis=1)

    search_test_dataset_df['candidate_answer_ratio_round'] = search_test_dataset_df['candidate_answer_ratio'].round(2)

    plt.figure(figsize=(19, 6))
    plt.xticks(rotation=45)
    sns_plot = sns.countplot(data=search_test_dataset_df, x='candidate_answer_ratio_round')
    plt.savefig(f'{config["REPORT_PATH"]}/candidate_answer_ratio_round.png')


if __name__ == '__main__':
    main()
