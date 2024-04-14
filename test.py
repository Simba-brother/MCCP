

def test1():
    import pandas as pd
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_copy = df.copy()
    df_copy["A"] = [0,0,0]
    print(df.head())
    print(df_copy.head())

def test2():
    for i in range(5,10):
        print(i)
    
def test3():
    l1 = [1,2,3]
    l2 = [9,9,9]
    for a,b in  zip(l1,l2):
        print(a,b)
if __name__ == "__main__":
    test3()
    