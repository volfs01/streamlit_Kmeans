from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st 
import pandas as pd
from datetime import date,datetime
import os
from PIL import Image
import numpy as np
def main() :
    st.title('K-Means 클러스터링')

    # 1. csv 파일을 업로드 할 수 있다.
    

    st.title('파일 업로드 ')

    
    st.subheader('csv파일업로드')
    file = st.file_uploader('csv파일업로드 ', type='csv')
    
    if file is not None :
        df = pd.read_csv(file)  
        df = df.dropna()  
        st.dataframe(df)
        
    
    df = df.dropna()
       
        
    columns = df.columns
    select = st.multiselect('X로 사용할 컬럼을 선택하시오' ,columns)
    
    
    from sklearn.preprocessing import LabelEncoder , OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    X = [ ]
    
    X = []
    for data in select :
        if df[data].nunique() > 0 :
            if type(df.loc[ :,data][4]) == type(df['points'][4]) or type(df.loc[ :,data][4]) == type(df['price'][4]) :
                pass
            else :
                encoder  = LabelEncoder()
                X.append(encoder.fit_transform(df[data]))
            
        elif df[data].nunique() > 2 :
            if type(df.loc[ :,data][4]) == type(df['points'][4]) or type(df.loc[ :,data][4]) == type(df['price'][4]) :
                pass
            else :
                ct = ColumnTransformer( [ ( 'encoder' ,OneHotEncoder() , [0] ) ] , remainder ='passthrough')
                X.append(ct.fit_transform(df[data]))
                
            
        st.dataframe(X)
    
    
    
    st.subheader('WCSS를 위한 클러스터링 개수를 선택하시오')
    clustering_counts = st.slider('최대 그룹 선택' , 2,20 ,value=10)
    
    wcss = []
    for k in np.arange(1 ,clustering_counts+1) : 
        kmeans = KMeans(n_clusters= k  , random_state=5 )
        kmeans.fit(X)
        wcss.append( kmeans.inertia_)
    # st.write(wcss)
    x = np.arange(1, clustering_counts + 1)
    fig = plt.figure()
    plt.plot ( x ,wcss)
    plt.title ('The Elbow Method')
    st.pyplot(fig)
    
    # 실제로 그룹핑할 개수 선택 
    # k = st.sidebar('그룹 개수 결정' ,1,clustering_counts +1 )
    
    k = st.number_input('그룹 개수 결정' ,1 ,clustering_counts + 1)
    
    kmeans = KMeans(n_clusters= k , random_state= 5)
    y_pred = kmeans.fit_predict(X)
    df['Group'] = y_pred
    st.dataframe(df)
    
    df.to_csv('result.csv')
if __name__ == '__main__' :
    main()