import os
import logging
import pandas as pd

#preprocess data - handle missing values
class DataPreprocess:
    def __init__(self,df)-> pd.DataFrame:
        self.df=df
        
    def data_transformation(self,columns=['bed','bath','house_size','price']):
        self.columns=columns
        for i in self.df[self.columns]:
            if (self.df[i].dtype) == 'float64':
                self.df[i]=self.df[i].astype('int32')
                
    def feature_selection(self,drop_columns=['status']):
        self.drop_columns=drop_columns
        self.df=self.df.drop(columns=drop_columns)
        
    def x_y_split(self,)
        
        
        
        

#drop unnecessary features

#transform data