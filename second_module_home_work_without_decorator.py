class MyCountVectorizer:
    def __init__(self):
        self._feature_names = []
        self._matrix = []

    def drop_data(self):
        self._feature_names = []
        self._matrix = []

    def fit_transform(self, corpus):
        self.drop_data()
        data = map(lambda x: x.split(), corpus)
        names_map = {}
        for line in data:
            local_names = {}
            for word in line:
                if not names_map.get(word.lower()):
                    names_map[word.lower()] = True
                    self._feature_names.append(word.lower())

                local_names[word.lower()] = local_names.get(word.lower(), 0) + 1
            self._matrix.append(local_names)

        self._feature_names.sort()
        self._matrix = [[line.get(word, 0) for word in self._feature_names] for line in self._matrix]
        return self._matrix

    def get_feature_names(self):
        return self._feature_names
