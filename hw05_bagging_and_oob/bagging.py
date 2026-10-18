import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indices = self.indices_list[bag]
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = [model.predict(data) for model in self.models_list]
        return np.mean(predictions, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        if self.data is None:
            raise ValueError('OOB predictions require fitting with oob=True.')
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for model, train_indices in zip(self.models_list, self.indices_list):
            # Какие объекты модель видела при обучении
            seen_indices = set(train_indices)
            # Какие объекты она не видела
            oob_indices = [
                i for i in range(len(self.data))
                if i not in seen_indices
            ]
            if not oob_indices:
                continue
            predictions = model.predict(self.data[oob_indices])
            for index, prediction in zip(oob_indices, predictions):
                list_of_predictions_lists[index].append(prediction)
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([
            np.mean(predictions) if len(predictions) > 0 else None
            for predictions in self.list_of_predictions_lists
        ], dtype=object)
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        errors = []
        for target, prediction in zip(self.target, self.oob_predictions):
            if prediction is not None:
                errors.append((target - prediction) ** 2)

        if not errors:
            raise ValueError('No objects have OOB predictions.')
        return np.mean(errors)
