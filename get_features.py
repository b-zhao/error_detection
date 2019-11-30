import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import Phrases

def convert_onehot(df, feature):
    one_hot = pd.get_dummies(df[feature], prefix=feature)
    # concat to the data frame
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop([feature], 1)
    return df

def dict_fun(key):
    try:
        return(model[key])
    except:
        return([0]*20)

def convert_word2vec(df, feature, train=False):
    model = Word2Vec.load("word2vec.model")
    if train:
        print("start training")
        model.train([df[feature]], total_examples=1, epochs=1)
        print("finished training")
    df[feature] = df[feature].apply(dict_fun)
    # print(df[feature][0:10])
    return df


''' Drop columns where more than half records are NA. Then drop rows where at least one 
    attribute is NA. Return the resulting dataframe, dropped columns and dropped rows. '''
def remove_na(data_file):
    df = pd.read_csv(data_file)
    print("Before remove_na: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    
    drop_col = df.loc[:,df.isnull().sum(axis = 0) > df.shape[0]/2]
    df = df.dropna(axis='columns', thresh=df.shape[0]/2)   
    drop_row = df.loc[df.isnull().any(axis=1)]
    df = df.dropna()
    
    # print number of nulls at each column
    # num_na_col = df.isnull().sum(axis = 0)
    # print(num_na_col)  
    
    print("After remove_na: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    return df, drop_col, drop_row


''' Remove rows with non-numeric value for an attribute where >0.995 rows have numeric values.
    Return df, removed_rows, affect_columns''' 
def remove_type_errors(df):
    affect_columns = []
    removed_rows = []
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number): 
            mask = pd.to_numeric(df[col], errors='coerce').isna()
            # print(col, "number of non-numeric values: ", mask.sum())
            if mask.sum() < 0.005 * df.shape[0]:
                removed_rows.append(df[col].apply(lambda x: not x.isnumeric()))
                affect_columns.append(col)
                df = df[df[col].apply(lambda x: x.isnumeric())]
                df.loc[:][col] = pd.to_numeric(df[col], errors='coerce')
    print("After remove_type_errors: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    return df, removed_rows, affect_columns


''' Numerical columns remain the same. Categorical columns are one-hot encoded. Texts are 
    encoded using word2vec. Return resulting data frame. ''' 
def convert_features(df):
    bigram_transformer = Phrases(common_texts)
    model = Word2Vec(bigram_transformer[common_texts], min_count=1)
    model.save("word2vec.model")
    
    for col in df.columns:
        num_unique = len(df[col].unique())
        if np.issubdtype(df[col].dtype, np.number): 
            # numerical
            # print(col, "[numerical", "#unique =", num_unique, "]")
            continue
        elif num_unique < 60 or (num_unique < 0.01 * df.shape[0] and num_unique < 100): 
            # categorical
            # print(col, "[categorical", "#unique =", num_unique, "]")
            df = convert_onehot(df, col)        
        else:
            # text
            # print(col, "[text", "#unique =", num_unique, "]")
            df = convert_word2vec(df, col)
              
    print("After get_features: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    return df


''' Get feature vectors from a csv file. '''
def get_feature_vecs(filename):
    df, _, _ = remove_na(filename)
    df, _, _ = remove_type_errors(df)
    df = convert_features(df)
    return np.array(df)

# X = get_feature_vecs('data/hospital.csv')
X = get_feature_vecs('data/people.csv')

