import pandas as pd
import config

if __name__ == '__main__':
    df = pd.read_csv(f"{config.CONFIG['output_dir']}/results.csv")
    print(df)