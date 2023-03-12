import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

while True:
    print("\n[ Student ID: __ ]")
    print("[ Name : __ ]\n")
    init_input = int(input("1. Market Basket Analysis \n2. Wine Clustering \n3. Quit \n Enter number : "))
    if init_input == 1:
        algorithm = input("Select the algorithm ((a)priori or (f)p-growth : ")
        min_sup = float(input("Enter the minimum support: "))

        dataset = pd.read_csv(r"./groceries - groceries.csv")
        dataset = dataset.drop(columns = ['Item(s)'])
        data = [[x for x in i if not pd.isnull(x)] for i in dataset.to_numpy()]

        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        if algorithm in ["a", "apriori", "Apriori", "A", "(a)priori"]:
            result= apriori(df, min_support=min_sup, use_colnames = True)
            print(result)
            
        elif algorithm in ["f", "fp-growth", "fp", "F", "Fp-growth", "(f)p-growth"]:
            result = fpgrowth(df, min_support=min_sup, use_colnames= True)
            print(result)
        
        else :
            print("please select a or f")

    elif init_input == 2:
        algorithm = input("Select the algorithm ((k)-means or (h)ierarchical : ")
        num_clusters = int(input("Input the number of clusters: "))

        dataset = pd.read_csv(r"./wine-clustering.csv")

        if algorithm in ["k", "k-means", "K", "K-means", "(k)-means"]:
            kmeans_model = KMeans(n_clusters = num_clusters, random_state=0)
            pred = kmeans_model.fit_predict(dataset)
            print(pred)
        
        elif algorithm in ["h", "(h)ierarchical" "hierarchical", "H"]:
            hirar_model = AgglomerativeClustering(n_clusters = num_clusters)
            pred = hirar_model.fit(dataset)
            print(pred.labels_)
        else :
            print("please select k or h")

    elif init_input == 3:
        quit()
    
    else :
        print("please select 1 or 2 or 3")