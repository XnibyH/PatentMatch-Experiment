import pandas as pd

if __name__ == "__main__":
        # load training data
        df = pd.read_parquet('data/train.parquet')

        # name Unnamed: 0 column -> unnamed_col
        df.columns = ['unnamed_col', 'index', 'claim_id', 'patent_application_id', 'cited_document_id', 'text', 'text_b', 'label', 'date', 'DIznQ_0']

        # fix indexing
        # replace NaNs in columns 'index' and 'unnamed_col', change float to int
        df['index'] = df['index'].fillna(0).apply(lambda x: int(x))
        df['unnamed_col'] = df['unnamed_col'].fillna(0).apply(lambda x: int(x))

        # create updated_index assuming that unnamed_col and index are corresponding to each other
        df['updated_index'] = df.apply(lambda row: row['index'] if row['index'] != 0 else row['unnamed_col'], axis=1)

        # drop columns: unnamed_col, index and rearrange columns order
        df = df[[
                # 'unnamed_col',
                # 'index',
                'updated_index',
                'text',
                'text_b',
                'label',
                'claim_id',
                'patent_application_id',
                'cited_document_id',
                'date',
                'DIznQ_0',
                ]
                ]

        # drop NaN in subset 'text', 'text_b', 'label';
        # left NaN in other columns ('claim_id', 'patent_application_id', 'cited_document_id', 'date', 'DIznQ_0')
        # in purpose to save as much training data as possible
        df = df.dropna(subset=['updated_index', 'text', 'text_b', 'label'])

        # drop duplicated
        df = df.drop_duplicates(subset=['text', 'text_b', 'label'], keep='first')

        # measure length in chars for text and text_b
        df['text_len'] = df['text'].apply(lambda x: len(x))
        df['text_b_len'] = df['text_b'].apply(lambda x: len(x))

        # rename column updated_index to index
        df.columns = ['index', 'text', 'text_b', 'label', 'claim_id', 'patent_application_id', 'cited_document_id', 'date', 'DIznQ_0', 'text_len', 'text_b_len']

        # Train Dataset Information
        print("\nTrain Dataset Information")
        print(f"Number of samples: {len(df)}")
        print(f"Distinct patent applications: {df['patent_application_id'].nunique()}")
        print(f"Distinct cited documents: {df['cited_document_id'].nunique()}")
        print(f"Distinct claim texts: {df['text'].nunique()}")
        print(f"Distinct cited paragraphs: {df['text_b'].nunique()}")
        print(f"Median claim length (chars): {df['text_len'].median()}")
        print(f"Median paragraph length (chars): {df['text_b_len'].median()}")
        print(f"Mean claim length (chars): {int(df['text_len'].mean())}")
        print(f"Mean paragraph length (chars): {int(df['text_b_len'].mean())}")
        print(f"Labels - 0, Non-novelty-destroying: {len(df[df['label'] == 0])}")
        print(f"Labels - 1, Novelty-destroying: {len(df[df['label'] == 1])}")

        # Save Train Data
        df_train = df[['index', 'text', 'text_b', 'label']]
        df_train.set_index('index', drop=True, inplace=True)
        df_train.to_parquet('data/train_clean.parquet')
        print('\nTrain Dataset saved: data/train_clean.parquet')
        print(df_train.head())


        # Test Dataset
        df = pd.read_parquet('data/test.parquet')

        # dropna and drop_duplicates
        df = df.dropna(subset=['index', 'text', 'text_b', 'label']).drop_duplicates(subset=['text', 'text_b', 'label'])

        # fix indexing
        df['index'] = df['index'].apply(lambda x: int(x))

        # change label datatype to int
        # df['label'] = df['label'].apply(lambda x: int(x))

        # rearrange columns order
        df = df[[
                'index',
                'text',
                'text_b',
                'label',
                'claim_id',
                'patent_application_id',
                'cited_document_id',
                'date',
                'DIznQ_0',
                ]
                ]

        # measure length in chars for text and text_b
        df['text_len'] = df['text'].apply(lambda x: len(x))
        df['text_b_len'] = df['text_b'].apply(lambda x: len(x))

        # Test Dataset Information
        print("\nTest Dataset Information")
        print(f"Number of samples: {len(df)}")
        print(f"Distinct patent applications: {df['patent_application_id'].nunique()}")
        print(f"Distinct cited documents: {df['cited_document_id'].nunique()}")
        print(f"Distinct claim texts: {df['text'].nunique()}")
        print(f"Distinct cited paragraphs: {df['text_b'].nunique()}")
        print(f"Median claim length (chars): {df['text_len'].median()}")
        print(f"Median paragraph length (chars): {df['text_b_len'].median()}")
        print(f"Mean claim length (chars): {int(df['text_len'].mean())}")
        print(f"Mean paragraph length (chars): {int(df['text_b_len'].mean())}")
        print(f"Labels - 0, Non-novelty-destroying: {len(df[df['label'] == 0])}")
        print(f"Labels - 1, Novelty-destroying: {len(df[df['label'] == 1])}")

        # Save Test Data
        df_test = df[['index', 'text', 'text_b', 'label']]
        df_test.set_index('index', drop=True, inplace=True)
        df_test.to_parquet('data/test_clean.parquet')
        print('\nTest Dataset saved: data/test_clean.parquet')
        print(df_test.head())

        # check for duplicated index in Train and Test sets
        print('\nChecking for duplicated indexes in datasets...')
        check_indexes_list = df_train.index.tolist()
        check_indexes_list.extend(df_test.index.tolist())

        if len(check_indexes_list) == len(set(check_indexes_list)):
        print("No duplicated index found.")
        else:
        print('Found duplicated index!')
