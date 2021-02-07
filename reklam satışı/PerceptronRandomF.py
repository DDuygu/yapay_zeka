import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split # Öğrenme modülü
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from imblearn.metrics import sensitivity_score
from sklearn.preprocessing import LabelEncoder
#LabelEncoder : Elimizdeki verileri direk sayısal temsillerine dönüştürmeye yarar.

#AMAÇ: satışa etkisi değişkenini tahmin için perceptron ve Random Forest kullanıldı. 
#Hangisinin ne kadar başarılı olduğunu gösteren accuracy, precision, sensitivity değerleri gösterildi.  

df = pd.read_csv('Bveriler.csv')

df.index = df.iloc[:,0]   #sektördeki tüm verileri index atadık.
df.index.names=[''] # isim kaldırıldı


label_encoder = LabelEncoder() 

# Dönüşüm gerçekleşir..
df['Sektör']= label_encoder.fit_transform(df['Sektör'])

print(df)
print("")


X = df.iloc[:,0:9]  # 0 dan 9 a kadar olanı X e ata.
y = df.iloc[:,9]   #9 uncu olanıda y ye ata.

print("******************PERCEPTRON********************")
print("")
# Kullanılan fonksiyon train_test_split dir.
# Veriyi 2 ye ayırır. %80 ni ögrenme(train)için %20 si ise test için kullanılacaktır..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape) # kaç tane sutun ve veriden oluştugunu ögreniriz
print("")
print("")
clf=Perceptron(max_iter=40, random_state=0) # perceptron kuruyoruz
print(clf)
print("")
clf.fit(X_train, y_train)    # x train ve y traine göre ögrenmeyi uygula..
y_pred = clf.predict(X_test)  # x testin y testlerini tahmin et..
print(y_pred)
print("")
print(clf.predict([[58,543,14,250,135,18,61,776,9]])) #tahmin degerleri
print(y_pred)
print("")
print("Accuracy Score=", accuracy_score(y_test, y_pred))
print("Precision Score=", precision_score(y_test, y_pred, average='macro'))
print("Sensitivity Score=", sensitivity_score(y_test, y_pred, average='macro'))
print("")
print("******************RANDOM FOREST********************")
print("")
print("")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape) # kaç tane sutun ve veriden oluştugunu ögreniriz
print("")
print("")
clf = RandomForestClassifier(n_estimators=100, max_depth= 2, random_state=0)  # RandomForestClassifier kuruyoruz..
clf.fit(X,y) 
print("")
print("")
print("")
clf.fit(X_train, y_train)   # x train ve y traine göre ögrenmeyi uygula..
y_pred = clf.predict(X_test)   # x testin y testlerini tahmin et..
print(clf.feature_importances_)   # Her bir sütunun önem derecesi yazılır.
print("")
print("")
print("")
print(clf.predict([[14,89,40,16,23,8,191,76,578]])) #tahmin degerleri
print(y_pred)
print("")
print("Accuracy Score=", accuracy_score(y_test, y_pred))  # sistem doğruluk oranı
print("Precision Score=", precision_score(y_test, y_pred, average='macro'))
print("Sensitivity Score=", sensitivity_score(y_test, y_pred, average='macro'))



