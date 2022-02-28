from cs506.sim import euclidean_dist
class KNN():
    def __init__(self, points, labels, k) -> None:
        """
        Initialise a KNN classifier.
        - `points`: Points in the dataset.
        - `labels`: Labels correspoding to each point.
        - `k`: K-nearest neighbours to choose from.
        """
        if k < 1 or k > len(points):
            raise ValueError('k needs to be in range!')
        self.__points = points
        self.__labels = labels
        self.__k = k
        return

    def __getNeighbours(self, test_point):
        """
        Get the K nearest neighbours of a test point.
        """
        distances = []
        for point, label in zip(self.__points, self.__labels):
            distances.append(
                (euclidean_dist(point, test_point), point, label)
            )

        distances.sort(key=lambda x : x[0])

        neighbours = []
        for i in range(self.__k):
            neighbours.append(distances[i][1:])
        return neighbours

    def __predictClass(self, neighbouring_classes, method='Majority Vote'):
        """
        Aggregation method to predict a class from the `k` nearest neighbours.
        :param neighbouring_classes - The k nearest classes.
        :return class - The class of the test point.
        """
        if method == 'Majority Vote':
            return max(set(neighbouring_classes), key=neighbouring_classes.count)
        else:
            raise ValueError('Aggregation method needs to be selected from the list!')

    def predict(self, test_point, method="Majority Vote"):
        """
        Predict the class of the test point given the `k` nearest neighbours.
        :param test_point Predict the class of this point.
        :return class Which class does this point belong to.
        """
        neighbours = self.__getNeighbours(test_point)
        classes = [point[-1] for point in neighbours]
        return self.__predictClass(classes, method)
