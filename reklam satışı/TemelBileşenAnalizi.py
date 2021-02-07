import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#LabelEncoder : Elimizdeki verileri direk sayısal temsillerine dönüştürmeye yarar..

#AMAÇ:PCA algoritmasıyla verileri (satışa etkisi değişkeni hariç tutularak) iki değişkene indirildi ve bu haliyle grafiği çizildi. 

df = pd.read_csv('Bveriler.csv')

df.index = df.iloc[:,0]   #sektördeki tüm verileri index atadık.
df.index.names=[''] # isim kaldırıldı


label_encoder = LabelEncoder() 

# Dönüşüm gerçekleşir..
df['Sektör']= label_encoder.fit_transform(df['Sektör'])

print(df)
print("")

X = df.iloc[:,0:9]   #satışa etkisi değişkeni hariç 



pca = PCA(n_components=2)   #2 bileşenli bir pca nesnesi yaratıldı.
pca.fit(X)   #pca uygulandı..

explained_variance = pca.explained_variance_ratio_   #"Açıklanan varyans oranı" nı bulmak için explained_variance_ratio_ kullanırız
print("***Açıklanan Varyans Oranı***")
print(pca.explained_variance_ratio_)       
print("")
print("***Tekil Degerleri***")
print(pca.singular_values_)   
print("")
print("")

# IncrementalPCA (Buyuk Veri Seti için kullanılır) 
ipca = IncrementalPCA(n_components=2, batch_size=3)  
ipca.fit(X)    #ipca yı uygula

print(ipca.transform(X))  
print("")


#ÇİZİM
z=ipca.transform(X) # pca de x veri setini dönüştürüp, değişkene atarız.

plt.scatter(z[:,0], z[:,1], s=7, cmap="aqua")   #plt tan scatter diyagramını getiriyoruz


plt.title("Temel Bileşen Analizi (PCA)")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

