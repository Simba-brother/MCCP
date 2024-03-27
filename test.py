def f1():
    print(a)

def test1():
    import pandas as pd
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_copy = df.copy()
    df_copy["A"] = [0,0,0]
    print(df.head())
    print(df_copy.head())
    
if __name__ == "__main__":
    test1()
    