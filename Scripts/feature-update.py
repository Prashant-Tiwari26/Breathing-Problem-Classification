import pandas as pd

def process():
    df = pd.read_csv("Data/Processed/processed_metadata.csv", index_col=False)
    train = pd.read_csv("Data/Processed/train_features.csv", index_col=False)
    test = pd.read_csv("Data/Processed/test_features.csv", index_col=False)
    val = pd.read_csv("Data/Processed/val_features.csv", index_col=False)

    for column in train.columns:
        df[column].replace(
            {
                True:1,
                False:0
            },
            inplace=True
        )
        train[column].replace(
            {
                True:1,
                False:0
            },
            inplace=True
        )
        test[column].replace(
            {
                True:1,
                False:0
            },
            inplace=True
        )
        val[column].replace(
            {
                True:1,
                False:0
            },
            inplace=True
        )

    df.to_csv("Data/Processed/processed_metadata.csv", index=False)
    train.to_csv("Data/Processed/train_features.csv", index=False)
    test.to_csv("Data/Processed/test_features.csv", index=False)
    val.to_csv("Data/Processed/val_features.csv", index=False)

if __name__ == '__main__':
    process()