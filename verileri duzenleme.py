# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from itertools import cycle,islice
import os


#csv dosyalarimizi yukulyoruz
data1= pd.read_csv("ISIC_2019_Training_Metadata.csv")
data2= pd.read_csv("ISIC_2019_Training_GroundTruth.csv")
data1.shape


modifydata1=data1.drop(['lesion_id'], axis=1, inplace=True)
#
#print(modifydata1)

#modifydata1=data1.dropna()
modifydata1 = data1.dropna(subset=['age_approx', 'anatom_site_general', 'gender'])# subset kismini yazmasakda olur
modifydata1= modifydata1.reset_index(drop=True)
print(modifydata1)
#print(modifydata1.isnull().sum())



#print(data1)

#verileri birlestir
birlestirme = pd.merge(modifydata1, data2, on = 'image')
birlestirme.set_index('image', inplace = True)



#1 olani al
bircekilenDf = pd.DataFrame([x for x in np.where(data2 == 1,data2.columns,'').flatten().tolist()
    if len(x) > 0], columns= (["Sınıf_Adı"]))
birlestirilenDf = pd.concat([modifydata1, bircekilenDf], axis=1, join='inner')
#print(birlestirilenDf)

#son islem dosyam
birlestirilenDf.to_csv(r'/Users/onno/Desktop/MAKINEEE/SONISLEMDOSYAM.csv', index = False)



colors = ['r', 'g', 'b', 'c', 'm']
df= pd.read_csv("SONISLEMDOSYAM.csv")

#lezyon
#numbers = df["Sınıf_Adı"].value_counts()
#paths = df["Sınıf_Adı"].value_counts().keys()
#plt.title("lezyon")
#plt.ylabel('count')
#plt.bar(paths, numbers,color=colors)
#plt.savefig('lezyon.png')


#bölge
#numbers = df["anatom_site_general"].value_counts()
#paths = df["anatom_site_general"].value_counts().keys()
#plt.title("bolge")
#plt.ylabel('count')
#plt.bar(paths, numbers,color=colors)
#plt.xticks(rotation=90)

#plt.tight_layout()
#plt.savefig('anato.png')


#cinsiyet
#numbers = df["gender"].value_counts()
#paths = df["gender"].value_counts().keys()
#plt.title("cinsiyet")
#plt.ylabel('count')
#plt.bar(paths, numbers,color=colors)
#plt.savefig('cinsiyet.png')

#yas
#numbers = df["age_approx"].value_counts()
#paths = df["age_approx"].value_counts().keys()
#plt.title("Yas")
#plt.ylabel("count")
#plt.bar(paths, numbers,color=colors)
#plt.savefig(u"yas.png")


#melonama yaş grafik
#df[df["Sınıf_Adı"] == 'MEL'].groupby("gender")['image'].nunique().plot(kind='bar', color = colors)
#plt.xticks(rotation=0)
#plt.xlabel("Cinsiyet")
#plt.ylabel("Adet")
#plt.tight_layout()
#plt.show()


#melonama bolge
#df[df["Sınıf_Adı"]=='MEL'].groupby("anatom_site_general")['image'].nunique().plot(kind='bar', color = colors)
#plt.xlabel("Lezyon Bolgesi")
#plt.ylabel("Adet")
#plt.show()
#plt.tight_layout()
#plt.savefig("melonamaanatom.png")



