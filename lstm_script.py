import numpy as np 
import pandas as pd
from tqdm import tqdm

import data_process_lstm
from lstm_model import build_model_and_train

def get_dims(users, sessions_grouped, one_hot_representation, one_hot_representations_sessions):
    nb = 0
    max_len = 0
    for user in tqdm(users.fillna('nan').values):
        try:
            sessions_user = sessions_grouped.get_group(user[0])
            l = len(sessions_user)
            if nb == 0:
                u = data_process_lstm.data_processing_by_user_lstm(user, sessions_user, one_hot_representation, one_hot_representations_sessions)
                d = np.array(u.features).shape
            if max_len < l:
                max_len = l
            nb = nb + 1
        except KeyError:
            continue
    return nb, d, max_len

def get_users_and_labels(users, sessions_grouped, one_hot_representation, one_hot_representations_sessions, max_len, d):
    labels = []
    z = 0
    preprocessed_users = np.zeros((22000, max_len, d[1]))
    for user in tqdm(users.fillna('nan').values):
        try:
            sessions_user = sessions_grouped.get_group(user[0])
            u = data_process_lstm.data_processing_by_user_lstm(user, sessions_user, one_hot_representation, one_hot_representations_sessions, max_len=max_len)
            if not np.isfinite(u.features).all():
                print(user[0])
                break
            #print(len(features))
            labels.append(user[-1])
            preprocessed_users[z] = u.features
            z = z + 1
            if z >= 22000:
                break
        except KeyError:
            continue
    return preprocessed_users, labels

def train_test_data(preprocessed_users, labels):
    # permutation = np.random.permutation(len(preprocessed_users))
    # preprocessed_users = np.array(preprocessed_users)[permutation]
    # labels = np.array(labels)[permutation]
    l = len(preprocessed_users)
    x_train, y_train = preprocessed_users[:int(0.85 * l)], labels[:int(0.85 * l)]
    x_val, y_val = preprocessed_users[int(0.85 * l):], labels[int(0.85 * l):]
    return x_train, y_train, x_val, y_val

def get_one_hot_representation(categories, add_other=False):
    if add_other:
        categories = np.concatenate((categories, ['OTHER']))
    return dict(zip(categories, list(range(len(categories)))))

if __name__ == "__main__":
    sessions = pd.read_csv('./data/sessions.csv')
    
    print("parse categories session...")
    sessions['secs_elapsed'] = sessions['secs_elapsed'].fillna(0.0)
    sessions = sessions.fillna('nan')
    sessions_grouped = sessions.groupby(['user_id'])
    var = np.var(sessions['secs_elapsed'])
    mean = np.mean(sessions['secs_elapsed'])
    sessions['secs_elapsed'] = (sessions['secs_elapsed'] - mean) / var

    action_type = np.unique(sessions['action_type'].fillna('nan'), return_counts=True)
    action = np.unique(sessions['action'].fillna('nan'), return_counts=True)
    action_detail = np.unique(sessions['action_detail'].fillna('nan'), return_counts=True)
    devices = np.unique(sessions['device_type'].fillna('nan'), return_counts=True)
    action_deleted_rare = action[0][np.where(action[1] > 50)]
    one_hot_representations_sessions = {'action': get_one_hot_representation(action_deleted_rare, add_other=True), 
                                  'action_type': get_one_hot_representation(action_type[0]),
                                  'action_detail': get_one_hot_representation(action_detail[0]),
                                  'device_type': get_one_hot_representation(devices[0])}
    
    print("parse categories user...")
    users = pd.read_csv('./data/train_users_2.csv')
    gender = np.unique(users['gender'])[1:]
    signup_method = np.unique(users['signup_method'])
    signup_flow = np.unique(users['signup_flow'])
    affiliate_channel = np.unique(users['affiliate_channel'])
    affiliate_provider = np.unique(users['affiliate_provider'])
    language = np.unique(users['language'])
    first_affiliate_tracked = np.unique(users['first_affiliate_tracked'].fillna('nan'))
    signup_app = np.unique(users['signup_app'])
    first_device_type = np.unique(users['first_device_type'])
    first_browser, counts_first_browser = np.unique(users['first_browser'], return_counts=True)
    first_browser = first_browser[np.where(counts_first_browser > 50)[0]]

    one_hot_representation = {
        4: get_one_hot_representation(gender),
        6: get_one_hot_representation(signup_method),
        7: get_one_hot_representation(signup_flow),
        8: get_one_hot_representation(language),
        9: get_one_hot_representation(affiliate_channel),
        10: get_one_hot_representation(affiliate_provider),
        11: get_one_hot_representation(first_affiliate_tracked),
        12: get_one_hot_representation(signup_app),
        13: get_one_hot_representation(first_device_type),
        14: get_one_hot_representation(first_browser, add_other=True),
    }

    print("get dimensions")
    nb, d,max_len = get_dims(users, sessions_grouped, one_hot_representation, one_hot_representations_sessions)
    print(nb, ' ', ' ', d, ' ',max_len)
    print('preprocess...')
    preprocessed_users, labels = get_users_and_labels(users, sessions_grouped, one_hot_representation, one_hot_representations_sessions, max_len, d)

    print("process labels...")
    labels_base = users['country_destination']
    country_code = get_one_hot_representation(np.unique(labels_base))
    labels_encoded = []
    for i in labels:
        m = [0.0]*12
        m[country_code[i]] = 1.0
        labels_encoded.append(m)
    labels_encoded = np.array(labels_encoded)
    del labels_base
    x_train, y_train, x_val, y_val = train_test_data(preprocessed_users, labels_encoded)
    del preprocessed_users
    del labels_encoded
    del labels

    build_model_and_train(x_train, y_train, x_val, y_val, max_len, d)