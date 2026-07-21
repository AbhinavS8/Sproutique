import pandas as pd
df = pd.read_csv("Crop_recommendation (1).csv")
selected_column = ["temperature","humidity","ph"]
df_selected = df[selected_column]
df_selected.to_csv('data_reduced.csv', index=False)