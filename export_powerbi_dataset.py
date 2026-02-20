import pandas as pd
import os

data_path = "C:/Users/Vamshi/Desktop/Intrusion_detection/data/"

gan = pd.read_csv(f"{data_path}gan_detailed_results.csv")
dbn = pd.read_csv(f"{data_path}dbn_detailed_results.csv")

combined = pd.concat([gan, dbn], ignore_index=True)

output_file = f"{data_path}powerbi_intrusion_dashboard_dataset.csv"
combined.to_csv(output_file, index=False)

print("Power BI dataset created successfully.")
