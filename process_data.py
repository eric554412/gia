import pandas as pd
import numpy as np

def process_data(path1, path2, path3):
    '''Prcess and merge data from three CSV files.'''
    df1 = pd.read_csv(path1, encoding = 'UTF-16 LE', sep = '\t', parse_dates = ['年月'])
    df2 = pd.read_csv(path2, encoding = 'UTF-8')
    df1 = df1[df1['證券代碼'].isin(df2['證券代碼'].unique())]
    set1 = set(df1['證券代碼'].unique())
    set2 = set(df2['證券代碼'].unique())
    only_in_df2 = set2 - set1
    only_in_df1 = set1 - set2
    print(f"Only in df2: {only_in_df2}")
    df3 = pd.read_csv(path3, encoding = 'UTF-16 LE', sep = '\t', parse_dates = ['年月'])
    df = pd.concat([df1, df3], ignore_index=True)
    df = df.sort_values(by = ['證券代碼', '年月']).reset_index(drop = True)
    return df

if __name__ == "__main__":
    df = process_data("fund_roi_monthly.csv", "fund_data.csv", "missing_fund.csv")
    print(df.head())
    df.to_csv("merged_fund_data.csv", index = False)