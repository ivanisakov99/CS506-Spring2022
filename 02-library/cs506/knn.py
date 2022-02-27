class KNN():
    def __init__(self, points, labels, k) -> None:
        """
        Initialise a KNN classifier.
        - `points`: Points in the dataset.
        - `labels`: Labels correspoding to each point.
        - `k`: K-nearest neighbours to choose from.
        """
        if k < 1 or k > len(points):
            raise ValueError('k needs to be in range')
        self.__points = points
        self.__labels = labels
        self.__k = k
        return

    def __euclideanDistance(self, point1, point2):
        """
        Calculate the euclidean distance between 2 points.
        """
        dist = 0
        for i in range(len(point1)):
            dist += (point1[i] - point2[i])**2
        return dist**(1/2)

    def __getNeighbours(self, test_point):
        """
        Get the K nearest neighbours of a test point.
        """
        distances = []
        for point, label in zip(self.__points, self.__labels):
            distances.append(
                (self.__euclideanDistance(point, test_point), point, label)
            )

        distances.sort(key=lambda x : x[0])

        neighbours = []
        for i in range(self.__k):
            neighbours.append(distances[i][1:])
        return neighbours

    def predict(self, test_point):
        """
        Predict the class of the test point given the `k` nearest neighbours.
        :param test_point Predict the class of this point.
        :return class Which class does this point belong to.
        """
        neighbours = self.__getNeighbours(test_point)
        classes = [point[-1] for point in neighbours]
        return max(set(classes), key=classes.count)
