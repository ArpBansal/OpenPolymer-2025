import pandas as pd
from config import config
from data_processing import load_data, separate_subtables
from train import train_or_predict

def run_inference():
    train_df, test_df = load_data()
    subtables = separate_subtables(train_df)

    test_smiles = test_df['SMILES'].tolist()
    test_ids = test_df['id'].values
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    output_df = pd.DataFrame({'id': test_ids})
    MODEL_DIR1 = '/kaggle/input/neurips-2025/xgb_v3'
    train_or_predict(labels, subtables, test_smiles, test_ids, output_df, train_model=False, model_dir=MODEL_DIR1)
    print(output_df)
    output_df.to_csv('submission_xgb.csv', index=False)
    print("Saved submission_xgb.csv")

if __name__ == "__main__":
    run_inference()
