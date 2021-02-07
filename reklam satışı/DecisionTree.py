import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Sınıflandırma modülü
from sklearn.preprocessing import LabelEncoder

#LabelEncoder : Elimizdeki verileri direk sayısal temsillerine dönüştürmeye yarar ve 
#kategorik her veriye sayısal bir değer atar. 

#AMAÇ: Sadece, maliyet, süre ve sektör değişkenleri açısından bakıldığında 
#Satışa Etkisini karar ağacı algoritması ile makineye öğretildi ve Öğrenme sonuçları yazdırıldı.

df=pd.read_csv('Bveriler.csv')
print(df.head())
print("")

df.index = df.iloc[:,0]   #sektördeki tüm verileri index atadık.
df.index.names=[''] # isim kaldırıldı

# değişkene LabelEncoder fonksiyonu atanır..
label_encoder = LabelEncoder() 

# Dönüşüm gerçekleşir..
df['Sektör']= label_encoder.fit_transform(df['Sektör'])

print(df)
print("")

df['Satışa Etkisi']= label_encoder.fit_transform(df['Satışa Etkisi'])
print(df)

# yeni veri tabanı oluşturulur..
X_train = df.loc[:,'Sektör':'reklam maliyeti TL']
Y_train = df.loc[:,'Satışa Etkisi']
print(df.head(9))

tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0) # DecisionTreeClassifier kuruyoruz
print(tree)


# Modeli uygula
tree.fit(X_train, Y_train)


# model test edilir..
prediction = tree.predict([[20,47,89],[78,29,63],[87,98,122]])
print("")
print("****TAHMİN****")
print (prediction) 







