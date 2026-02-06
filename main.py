import pandas as pd
import glob

submission_files = ["submission_gnn.csv", "submission_xgb.csv"]
output_dfs = [pd.read_csv(f) for f in submission_files]

final_df = pd.concat(output_dfs, axis=0).groupby('id').mean().reset_index()
final_df.to_csv('submission.csv', index=False)
print("Saved final submission.csv")
print(final_df.head())