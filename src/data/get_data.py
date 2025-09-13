#src.data.get_data.py
from sklearn.datasets import load_iris
from src.config.settings import app_config

def get_data():
    print("🚀 Starting Stage: Data Ingestion")
    output_path = app_config.paths.raw_data
    print(f"   - Saving raw data to: {output_path}")
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)
    df.to_csv(output_path, index=False)

    print("   - ✅ Raw dataset saved successfully.")
    print("🏁 Finished Stage: Data Ingestion")

if __name__ == "__main__":
    get_data()

