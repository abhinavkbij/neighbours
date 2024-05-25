import math
from typing import List, Optional, Union
from collections import Counter
from abc import abstractmethod

class Neighbours:
    def __init__(self, k: int = 3, task: str = "classification"):
        self.k = k
        self.task = task

    def __check_data_types(self, X: List[List[Union[int, str, float]]], y: List[int]): ...

    def __find_neighbours(self, X: List[List[Union[int, float]]], datum: List[Union[int, float]]):
        # after sanity check e.g. data type checks iterate through data points and store the calculated distances from new point into a list

        distances = {}
        id = 0

        for x in X:
            distances[id] = Neighbours.calculate_mh_distance(x, datum)
            id += 1

        sorted_distances = dict(sorted(distances.items()), key=lambda item: item[1])

        knn_ids = list(sorted_distances.keys())[:self.k]
        print ("knn_ids: ", knn_ids)
        return knn_ids

    @abstractmethod
    def fit(self, X: List[List[Union[int, float]]], y: List[int]):
        self.X = X
        self.y = y

    @abstractmethod
    def predict(self,  datum: List[Union[int, float]]):
        knn_ids = self.__find_neighbours(self.X, datum)
        knn_labels = [self.y[id] for id in knn_ids]
        if self.task == "classification":
            knn = Counter(knn_labels)
            print (knn)
            for k in knn.keys():
                return k
        elif self.task == "regression":
            return sum(knn_labels)/len(knn_labels)

        # for regression
        # return knn

    @staticmethod
    def calculate_mh_distance(a: List[Union[int, float]], b: List[Union[int, float]]) -> Union[int, float]:
        assert len(a) == len(b)

        __distance = 0
        for i in range(len(a)):
            __distance += (a[i]-b[i])**2

        distance = math.sqrt(__distance)
        return distance



if __name__ == "__main__":
    neighbours = Neighbours(k=3)

    print (neighbours.calculate_mh_distance([1,2,3,4], [4,3,2,1]))

    X = [[j for j in range(i, i+10)] for i in range(100)]
    y = [0 if i%2 == 0 else 1 for i in range(100)]

    print (X, y)

    neighbours.fit(X, y)
    print (neighbours.predict([0,1,2,3,4,5,6,7,8,9]))
