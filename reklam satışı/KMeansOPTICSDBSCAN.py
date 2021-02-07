import numpy as np
import pandas as pd
from matplotlib import pyplot as plt #matplot kütüphanesinden pyplot gelir (grafik için)
from sklearn.cluster import KMeans  #kumelemeden KMeans i getirilir..
from sklearn.cluster import OPTICS 
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import LabelEncoder
#LabelEncoder : Elimizdeki verileri direk sayısal temsillerine dönüştürmeye yarar..

#AMAÇ: Bu veri setine göre 4 tür reklam vardır,
#(satışa etkisi değişkenini hariç tutularak) K-Means, Optics ve DBSCAN algoritmalarını kullanarak kümeleme yapıldı. 
#Her bir algoritmanın kümeleri grafik halinde gösterildi.

df = pd.read_csv('Bveriler.csv')

df.index = df.iloc[:,0]   #sektördeki tüm verileri index atadık.
df.index.names=[''] # isim kaldırıldı


label_encoder = LabelEncoder() 

# Dönüşüm gerçekleşir..
df['Sektör']= label_encoder.fit_transform(df['Sektör'])

print(df)
print("")
print("")
print("******KMEANS******")

X = df.iloc[:,0:9]  

k=4      # 4 tane kumeye ayrıldı..
kmeans =KMeans(n_clusters=k)   #model Kuruldu.. n_clusters: parametrelerinden biri k dır.
k_fit=kmeans.fit(X)          #model Çalıştırıldı..

labels = kmeans.labels_    # KMeans i işliyoruz..yani tüm kayıtların hangi kümeye ait oldugu gösterilir..
centroids = kmeans.cluster_centers_   #belirlenen kume merkezlerine ulaşılır.

x_test = [[40, 7568, 18, 46, 32, 145, 69, 27, 264],
          [57, 27448, 95, 69, 611, 173, 85, 641, 367]]    #Yeni Verilerle Deneme Yapılır..

prediction= kmeans.predict(x_test)        #x_test tahmin edilir..
print("Tahmin:", prediction)     # Tahmini çalıştırıp, görürüz.
print("")
print("Merkezler:", centroids)  # Merkezi çalıştırıp, görürüz.
print("")
print("Etiketler:", labels)     # Etiketleri çalıştırıp, görürüz.


#ÇİZİM
colors = ['red','blue','green','purple'] #renkler
y = 0  # y ekseni tanımlandı. her zaman 0 dır..

for x in labels:
    plt.scatter(df.iloc[y,1], df.iloc[y,2],color=colors[x]) 
    y+=1  # y yi artıralımki sürekli dolaşsın..
    
for x in range(k):     
    lines = plt.plot(centroids[x,0],centroids[x,1],'kx')    #merkezler çizilir..
    plt.setp(lines,ms=15.0)  #kx in buyuklugu
    plt.setp(lines,mew=5.0)  #kx in kalınlığı
    
title = ('Kume Sayisi (k) = {}').format(k)
plt.title(title)
plt.xlabel('eruptions (mins)')
plt.ylabel('waiting (mins)')
plt.show()



print("")
print("******DBSCAN******")

X = df.iloc[:,0:9]
y = df.iloc[:,9]


clustering = DBSCAN(eps=10, min_samples=3).fit(X,y)  # model kuruldu..

print("")
print('****************************')
print(clustering.labels_)  #burda etiketleri söyleriz yani bu kaçıncı küme şeklinde..
print("")
print("")
d=clustering.labels_
print('****************************')
print(clustering)   #istatistikleri.. Yani bize DBSCAN in ne kullandığı hakkında bilgi verir.
print("")
print("")
plt.scatter(df.iloc[:,1], df.iloc[:,2],c=d, s=35, cmap='coolwarm')
plt.show()
print("")


print("******OPTICS******")

X = df.iloc[:,0:9]
y = df.iloc[:,9]

clustering = OPTICS( min_samples=2).fit(X,y)   # OPTICS kuruldu..

print(clustering.labels_)   # burda etiketleri söyleriz yani bu kaçıncı küme şeklinde..
print("")
print("************")
z=clustering.labels_   
print("")
print("")
print(clustering)    # OPTICS in ne kullandığı hakkında bilgi verir.

#ÇİZİM...
plt.scatter(df.iloc[:,1], df.iloc[:,2],c=z, s=35, cmap='Oranges') 
plt.show()
