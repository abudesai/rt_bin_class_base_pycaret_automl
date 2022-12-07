import os
import sys

import algorithm.model.classifier as classifier
import algorithm.utils as utils
import numpy as np

# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.data_schema = data_schema

    def _get_model(self):
        try:
            self.model = classifier.load_model(self.model_path)
            return self.model
        except:
            print(
                f"No model found to load from {self.model_path}. Did you train the model first?"
            )
        return None

    def predict(self, data, data_schema=None):
        # preprocessor = self._get_preprocessor()
        model = self._get_model()

        # if preprocessor is None:
        #     raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # make predictions
        preds = model.predict(data)
        # inverse transform the predictions to original scale
        # preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)
        # get the names for the id and prediction fields
        id_field_name = self.data_schema["inputDatasets"][
            "binaryClassificationBaseMainInput"
        ]["idField"]
        # return te prediction df with the id and prediction fields
        preds_df = data[[id_field_name]].copy()
        preds_df["prediction"] = preds.values

        return preds_df

    def predict_proba(self, data):
        preds = self._get_predictions(data)
        # get the name for the id field
        id_field_name = self.data_schema["inputDatasets"][
            "binaryClassificationBaseMainInput"
        ]["idField"]
        # return te prediction df with the id and class probability fields
        preds_df = data[[id_field_name]].copy()

        for c in preds.columns:
            preds_df[c] = preds[c]

        return preds_df

    def _get_predictions(self, data):
        model = self._get_model()
        preds = model.predict_proba(data)
        return preds
