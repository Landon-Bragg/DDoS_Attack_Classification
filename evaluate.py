import pandas as pd
import config

# This saves our results to a csv, could be good for final paper :D
if __name__ == '__main__':
    df = pd.read_csv(f"{config.CONFIG['output_dir']}/results.csv")
    print(df)