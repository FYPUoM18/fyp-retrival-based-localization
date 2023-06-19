from sklearn.neighbors import KDTree

class NNTrainer:

    def __init__(self,conf) -> None:
        self.conf=conf
    
    def trainNN(self,vectors):
    
        knn = KDTree(vectors)
        return knn

        

