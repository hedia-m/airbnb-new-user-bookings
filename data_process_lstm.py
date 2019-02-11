import numpy as np
import pandas as pd

elapsed_secs_column = 'secs_elapsed'
class data_processing_by_user_lstm(object):

    def get_one_hot_encoded_session(self, categories, ret):
        for name,one_hot in categories.items():
            cat_user = np.zeros((len(self.sessions), len(one_hot)))
            for i,value in enumerate(self.sessions[name]):
                v = float(self.sessions[elapsed_secs_column].values[i])
                if str(value) in one_hot:
                    cat_user[i][one_hot[str(value)]] =  v if np.isfinite(v) else 0.0
                else:
                    cat_user[i][one_hot['OTHER']] = v if np.isfinite(v) else 0.0
            ret = np.concatenate((ret, cat_user), axis=1)
        return ret

    def get_one_hot_encoded_user(self, categories):
        ret = []
        for i,one_hot in list(categories.items())[1:]:
            cat = [0.0]*len(one_hot)
            if self.user_row[i] in one_hot:
                cat[one_hot[self.user_row[i]]] = 1.0
            else:
                cat[one_hot['OTHER']] = 1.0
            ret = ret + cat
        return cat

    def get_one_hot_date(self, date, add_hour=False):
        if add_hour:
            timestamp = pd.to_datetime(date, format='%Y%m%d%H%M%S')
        else:
            timestamp = pd.to_datetime(date)
        month_one_hot = [0.0]*12
        month_one_hot[timestamp.month - 1] = 1.0
        day_one_hot = [0.0]*31
        day_one_hot[timestamp.day - 1] = 1.0
        dayweek_one_hot = [0.0]*7
        dayweek_one_hot[timestamp.weekday()] = 1.0
        if add_hour:
            hour_one_hot = [0.0]*24
            hour_one_hot[timestamp.hour] = 1.0
            return month_one_hot + day_one_hot + dayweek_one_hot + hour_one_hot
        return month_one_hot + day_one_hot + dayweek_one_hot

    def get_one_hot_age(self, age):
        one_hot_rep = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120]
        one_hot = [0.0] * len(one_hot_rep)
        if age == 'nan':
            return one_hot
        if age > 1900 and age < 2000:
            age = 2015 - age
        elif age > 2000:
            age = 4
        i = 0
        while i < len(one_hot_rep) and one_hot_rep[i] < age:
            i += 1
        if i == len(one_hot_rep):
            one_hot[i - 1] = 1.0
        else:
            one_hot[i] = 1.0
        return one_hot

    def __init__(self, user_row, sess, categories_row_user, categories, max_len=0):
        self.sessions = sess
        self.user_row = user_row
        features = []
        gender_one_hot = [0.0] * len(categories_row_user[4])
        if user_row[4] in categories_row_user[4]:
            gender_one_hot[categories_row_user[4][user_row[4]]] = 1.0
        features = features + gender_one_hot + self.get_one_hot_encoded_user(categories_row_user)
        features = features + self.get_one_hot_date(user_row[1])
        features = features + self.get_one_hot_date(user_row[2], add_hour=True)
        features = features + self.get_one_hot_age(user_row[5])
        f = np.array([features]*len(self.sessions))
        self.features = list(self.get_one_hot_encoded_session(categories, f))
        while len(self.features) < max_len:
            s = [0.0]*len(self.features[0])
            self.features.append(s)
        self.features = np.array(self.features)


        
        