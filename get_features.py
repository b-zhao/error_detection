import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import Phrases
import warnings

warnings.filterwarnings('ignore')


def convert_onehot(df, feature):
    one_hot = pd.get_dummies(df[feature], prefix=feature)
    # concat to the data frame
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop([feature], 1)
    return df

def convert_word2vec(df, feature, train=False):
    model = Word2Vec.load("word2vec.model")
    if train:
        model.train([df[feature]], total_examples=1, epochs=1)
    df[feature] = df[feature].apply(lambda x: model[x])
    return df


''' Drop columns where more than half records are NA. Then drop rows where at least one 
    attribute is NA. Return the resulting dataframe, dropped columns and dropped rows. '''
def remove_na(df):
    drop_col = df.loc[:,df.isnull().sum(axis = 0) > df.shape[0]/2]
    df = df.dropna(axis='columns', thresh=df.shape[0]/2)   
    drop_row = df.loc[df.isnull().any(axis=1)]
    df = df.dropna()
    
    # print number of nulls at each column
    # num_na_col = df.isnull().sum(axis = 0)
    # print(num_na_col)  
    
    return df, drop_col, drop_row


''' Remove rows with non-numeric value for an attribute where >0.995 rows have numeric values.
    Remove rows with numeric value for an attribute where >0.995 rows have non-numeric values.
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
                df.loc[:,col] = pd.to_numeric(df.loc[:,col], errors='coerce')
            if mask.sum() > 0.995 * df.shape[0] and mask.sum() < 1.0:
                removed_rows.append(df[col].apply(lambda x: x.isnumeric()))
                affect_columns.append(col)
                df = df[df[col].apply(lambda x: not x.isnumeric())]
    return df, removed_rows, affect_columns


''' Remove rows with a numerical cell that is not within threshold_quantile from the 
    column mean. '''
def remove_outliers_gaussian(df, threshold_quantile=0.9999):
    removed_rows = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number): 
            upper_bound = df[col].quantile(threshold_quantile)
            lower_bound = df[col].quantile(1 - threshold_quantile)
            removed_rows.append(df[col].apply(lambda x: x >= upper_bound or x <= lower_bound))
            df = df[df[col].apply(lambda x: x < upper_bound and x > lower_bound)]
    return df, removed_rows


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
            print(col, "[numerical", "#unique =", num_unique, "]")
            continue
        elif num_unique < 60 or (num_unique < 0.01 * df.shape[0] and num_unique < 100): 
            # categorical
            print(col, "[categorical", "#unique =", num_unique, "]")
            df = convert_onehot(df, col)        
        else:
            # text
            print(col, "[text", "#unique =", num_unique, "]")
            df = convert_word2vec(df, col)
              
    return df


''' Consider text columns only. Use word2vec to assign numerical values to each attribute for each
    record. Word2vec model is trained using records as documents and attribute values as words. 
    Return resulting data frame. ''' 
def convert_features2(df):
    text_cols = []
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number): 
            text_cols.append(col)
            
    X = np.array(df.loc[:,text_cols]).tolist()
    model = Word2Vec(X, min_count=1, size=10, workers=4)
    
    # replace each text by the vector (10 new numerical columns)
    new_df = pd.DataFrame()
    for col in text_cols:
        df[col] = df[col].apply(lambda x: model[x])
        vec_cols = df[col].apply(pd.Series)
        vec_cols.columns = [col + str(i) for i in range(10)]
        new_df = pd.concat([new_df, vec_cols], axis=1)

    for col in text_cols:
        df = df.drop([col], 1)
    df = pd.concat([df, new_df], axis=1)
    return df


''' Get feature vectors from a csv file. '''
def get_feature_vecs(filename):
    df = pd.read_csv(filename)
    print("Original table: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    
    df, _, _ = remove_na(df)
    print("After remove_na: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    
    df, _, _ = remove_type_errors(df)
    print("After remove_type_errors: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    
    df, _ = remove_outliers_gaussian(df)
    print("After remove_outliers_gaussian: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    
    df = convert_features2(df)
    print("After convert_features2: Number of records: ", df.shape[0], " :: Number of features: ", df.shape[1])
    
    return np.array(df)

# X = get_feature_vecs('data/hospital.csv')
X = get_feature_vecs('data/people.csv')

