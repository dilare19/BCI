# online/online_predict.py

class OnlinePredict:

    def __init__(self,model):

        self.clf = model['clf']

    def predict(self,X):

        prob = self.clf.predict_proba(X)[0]

        confidence = max(prob)

        pred = self.clf.predict(X)[0]

        return pred,confidence