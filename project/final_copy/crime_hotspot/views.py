from django.shortcuts import render
from scipy.sparse import data

from .classifier import model, transformer, find_spot

import json
import os

def index(request):
    return render(request, 'crime_hotspot/index_t.html')

def predict(request):
    params = {}
    try:
        params = {
            'address': request.POST['address'],
            'crime_type': request.POST['crime_type'],
            'model_type': request.POST['model_type']
        }
    except KeyError:
        return render(
            request,
            'crime_hotspot/index.html',
            {
                'params': 'error'
            }
        )
    else:
        X_train, X_test, y_train, y_test = model.read_data(
            params['crime_type'],
            os.path.join(os.getcwd(), 'crime_hotspot', 'classifier', 'mergedata')
        )
        assert len(X_train)==len(y_train), "LENGTH INCONSISTENT"
        address = json.loads(params['address'])
        params['address'] = address[0]
        dataframe = transformer.generate_input([address], params['crime_type'])
        outputs = {
            'y_pred': None,
            'f1': None,
            'nearest_10': [],
        }

        nearest_10 = find_spot.near_point(
            params['crime_type'],
            address[1],
            address[2],
            num=3
        )
        nearby_info = nearest_10['address'].values
        outputs['nearest_10'] = nearest_10[['label', 'lat', 'lng']]
        outputs['nearest_10'].loc[-1] = [0, address[1], address[2]]
        outputs['nearest_10'] = outputs['nearest_10'].to_dict('records')

        if params['model_type'] == "knn":
            _, y_pred, best_k = model.model_knn(X_train, dataframe, y_train)
            outputs['y_pred'] = y_pred[0]
            _, y_pred, best_k = model.model_knn(X_train, X_test, y_train)
            f1, _, _, _ = model.cal_score(y_test, y_pred)
            outputs['f1'] = f1
        elif params['model_type'] == 'dt':
            _, y_pred, best_depth = model.model_dt(X_train, dataframe, y_train, params['crime_type'])
            outputs['y_pred'] = y_pred[0]
            _, y_pred, best_depth = model.model_dt(X_train, X_test, y_train, params['crime_type'])
            f1, _, _, _ = model.cal_score(y_test, y_pred)
            outputs['f1'] = f1
        elif params['model_type'] == 'rf':
            _, y_pred= model.randomforest(X_train, dataframe, y_train)
            outputs['y_pred'] = y_pred[0]
            _, y_pred= model.randomforest(X_train, X_test, y_train)
            f1, _, _, _ = model.cal_score(y_test, y_pred)
        elif params['model_type'] == 'svm':
             _, y_pred = model.svm(X_train, dataframe, y_train)
             outputs['y_pred'] = y_pred[0]
             _, y_pred = model.svm(X_train, X_test, y_train)
             f1, _, _, _ = model.cal_score(y_test, y_pred)
        elif params['model_type'] == 'nb':
             _, y_pred = model.naivebayes(X_train, dataframe, y_train)
             outputs['y_pred'] = y_pred[0]
             _, y_pred = model.naivebayes(X_train, X_test, y_train)
             f1, _, _, _ = model.cal_score(y_test, y_pred)

        return render(
            request,
            'crime_hotspot/index_t.html',
            {
                'params': params,
                'prediction': outputs,
                'nearby_info': nearby_info
            }
        )