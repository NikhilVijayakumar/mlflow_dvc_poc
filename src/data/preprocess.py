#src.data.preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split


from src.config.settings import app_config

def main():

    print("🚀 Starting Stage: Data Preprocessing")

    # Load all paths and parameters from the central config
    input_path = app_config.paths.raw_data
    train_path = app_config.paths.train_data
    test_path = app_config.paths.test_data
    test_size = app_config.training.test_size
    random_state = app_config.training.random_state

    print(f"   - Loading raw data from: {input_path}")
    print(f"   - Splitting data with test_size={test_size} and random_state={random_state}")

    # Read the raw dataset
    df = pd.read_csv(input_path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['target'] if 'target' in df.columns else None
    )

    # Save the processed datasets to the paths defined in our config
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"   - ✅ Saved training data to: {train_path}")
    print(f"   - ✅ Saved testing data to: {test_path}")
    print("🏁 Finished Stage: Data Preprocessing")


if __name__ == "__main__":
    main()


