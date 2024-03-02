
import  pandas as pd

#  データをマージする
#　Laptopdata1.csvはutf-8格式
A = pd.read_csv('C:/Users/ZHOUHANGXU/Desktop/python/Laptopdata.csv', encoding='utf-8-sig')
B = pd.read_csv('C:/Users/ZHOUHANGXU/Desktop/python/master.csv', encoding='utf-8-sig')

A.columns = [col.replace('\ufeff', '') for col in A.columns]
B.columns = [col.replace('\ufeff', '') for col in B.columns]

table = A.merge(B,on='JANコード')


##データを処理
#delete
#del table['JANコード']
table = table.drop(['販売数量','SSD容量','ベンダー名','型番','品名','OS','BCNR初登場日','バッテリ駆動時間','アイテムコード','アイテム名','スペックカテゴリ','NBタイプ','形状','CPU','OS(詳細)','最大メモリ','メモリスロット(空き)','メモリスロット(最大)','GPU','ディスプレイ種類','モニタ解像度','HDD容量','光学ドライブ','バンドルソフト','通信機能','ワイヤレスLAN','その他ワイヤレスI/F','チューナー種類','HDMI端子(出)','PCカードスロット','メモリカードスロット','ICカードリーダ','バッテリ駆動時間(JEITA20)','幅','奥行','厚さ','カラーバリエーション','色系統','モデル','3D映像'], axis=1, inplace=False)

#name変更
colNameDict = {
    'JANコード':'JANCODE',
    '販売金額':'P',
    '日':'DATE',
    'クロック':'CLOCK',
    '標準メモリ':'MEMORY',
    'ディスプレイサイズ':'DISPLAYSIZE',
    'タッチパネル':'TOUCH1',
    '生体認証方式':'RECOG',
    '重量':'WEIGHT'
}
    
table.rename(columns = colNameDict, inplace=True)
    
#パソコンのタイプの数
unique_count = table.JANCODE.nunique()
print(unique_count)

#輸出特定の列
first = table.iloc[:4]
print(first)

#空値をキャンセルする
table = table.dropna(axis=0,how='any')

#density chartを作る
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(table['P']) #<1000000
sns.kdeplot(table['CLOCK']) #<3500
sns.kdeplot(table['MEMORY']) #<20000
sns.kdeplot(table['DISPLAYSIZE']) #>10
sns.kdeplot(table['WEIGHT']) #<3
plt.show()

table1 = table[table['P']<=1000000]
table1 = table1[table1['CLOCK']<=3500]
table1 = table1[table1['MEMORY']<=20000]
table1 = table1[table1['DISPLAYSIZE']>10]
table1 = table1[table1['WEIGHT']<3]
table = table1

unique_count = table.JANCODE.nunique()
print(unique_count)


##data処理
#単位の処理
table['P'] = table['P']/100000
table['CLOCK'] = table['CLOCK']/1000
table['MEMORY'] = table['MEMORY']/1000
table['DISPLAYSIZE'] = table['DISPLAYSIZE']/10

#lp: log(P)
import numpy as np
import math
table['LP'] = table['P'].apply(lambda x: math.log(x))

#date: D1-D24
table['D1'] = (table['DATE'] < 20200132).astype(int)
table['D2'] = ((table['DATE'] < 20200232) & (table['DATE']>=20200201)).astype(int)
table['D3'] = ((table['DATE'] < 20200332) & (table['DATE']>=20200301)).astype(int)
table['D4'] = ((table['DATE'] < 20200432) & (table['DATE']>=20200401)).astype(int)
table['D5'] = ((table['DATE'] < 20200532) & (table['DATE']>=20200501)).astype(int)
table['D6'] = ((table['DATE'] < 20200632) & (table['DATE']>=20200601)).astype(int)
table['D7'] = ((table['DATE'] < 20200732) & (table['DATE']>=20200701)).astype(int)
table['D8'] = ((table['DATE'] < 20200832) & (table['DATE']>=20200801)).astype(int)
table['D9'] = ((table['DATE'] < 20200932) & (table['DATE']>=20200901)).astype(int)
table['D10'] = ((table['DATE'] < 20201032) & (table['DATE']>=20201001)).astype(int)
table['D11'] = ((table['DATE'] < 20201132) & (table['DATE']>=20201101)).astype(int)
table['D12'] = ((table['DATE'] < 20201232) & (table['DATE']>=20201201)).astype(int)
table['D13'] = ((table['DATE'] < 20210132) & (table['DATE']>=20210101)).astype(int)
table['D14'] = ((table['DATE'] < 20210232) & (table['DATE']>=20210201)).astype(int)
table['D15'] = ((table['DATE'] < 20210332) & (table['DATE']>=20210301)).astype(int)
table['D16'] = ((table['DATE'] < 20210432) & (table['DATE']>=20210401)).astype(int)
table['D17'] = ((table['DATE'] < 20210532) & (table['DATE']>=20210501)).astype(int)
table['D18'] = ((table['DATE'] < 20210632) & (table['DATE']>=20210601)).astype(int)
table['D19'] = ((table['DATE'] < 20210732) & (table['DATE']>=20210701)).astype(int)
table['D20'] = ((table['DATE'] < 20210832) & (table['DATE']>=20210801)).astype(int)
table['D21'] = ((table['DATE'] < 20210932) & (table['DATE']>=20210901)).astype(int)
table['D22'] = ((table['DATE'] < 20211032) & (table['DATE']>=20211001)).astype(int)
table['D23'] = ((table['DATE'] < 20211132) & (table['DATE']>=20211101)).astype(int)
table['D24'] = ((table['DATE'] < 20211232) & (table['DATE']>=20211201)).astype(int)

#touch:touch1
table['TOUCH'] = (table['TOUCH1'] == '対応').astype(int)

#recog: 0 rface rfinger rboth
table['RNONE'] = (table['RECOG'] == 'なし').astype(int)
table['RFINGER'] = (table['RECOG'] == '指紋').astype(int)
table['RFACE'] = (table['RECOG'] == '顔').astype(int)
table['RBOTH'] = (table['RECOG'] == '指紋/顔').astype(int)

#table2:statistics:記述統計
sta = pd.DataFrame(table, columns=['P','CLOCK','MEMORY','DISPLAYSIZE','WEIGHT','TOUCH','RNONE','RFACE','RFINGER','RBOTH'])
a = sta.describe()
a = a.round(2) #小数点以下2桁に丸める
print(a)

#table3:時間によっての変化 
columns = ['P', 'CLOCK', 'MEMORY','DISPLAYSIZE','WEIGHT','TOUCH','RFACE','RFINGER','RBOTH']

for i in range(1, 25): #D1からD24まで繰り返す
  selected_data = table[table[f'D{i}'] == 1]
  print(i)
  print(selected_data.JANCODE.nunique()) #製品種類
  print(selected_data[columns].mean()) #averages

##5.回帰
##5.1 time dummy variable method
import statsmodels.api as sm

#5.1.1 時間ダミーを含める　すべての時間の回帰
fit=sm.formula.ols('LP~ CLOCK+MEMORY+DISPLAYSIZE+WEIGHT+TOUCH+RFACE+RFINGER+RBOTH+D2+D3+D4+D5+D6+D7+D8+D9+D10+D11+D12+D13+D14+D15+D16+D17+D18+D19+D20+D21+D22+D23+D24',data=table).fit()
print(fit.summary())

#5.1.2 時間ダミーを含める　相隣時間の回帰
for i in range(1, 24):
      selected_data = table[(table[f'D{i}'] == 1) | (table[f'D{i+1}'] == 1)]
      formula = f"LP ~ CLOCK + MEMORY + DISPLAYSIZE + WEIGHT + TOUCH + RFACE + RFINGER + RBOTH + D{i}"
      fit=sm.formula.ols(formula=formula,data=selected_data).fit()
      print(i)
      print(fit.summary())

#5.2 characterstics price index method
for i in range(1, 25):
      selected_data = table[table[f'D{i}'] == 1]
      formula = f"LP ~ CLOCK + MEMORY + DISPLAYSIZE + WEIGHT + TOUCH + RFACE + RFINGER + RBOTH"
      fit=sm.formula.ols(formula=formula,data=selected_data).fit()
      print(i)
      print(fit.summary())


##6 価格指数
##6.1 time dummy variable method
##6.1.1 すべての時間の回帰 D1-Dt
#5-1というCSV fileを作りました。delta1は5.1.1のmodelの時間ダミーの係数。delta2は5.1.2のmodelの時間ダミーの係数。

C = pd.read_csv('C:/Users/ZHOUHANGXU/Desktop/python/5-1.CSV', encoding='utf-8-sig')
C.columns = [col.replace('\ufeff', '') for col in C.columns]

#index1:
C['index1'] = np.exp(C['delta1'])
print(C.index1)


#index2:
for i in range(1, 24):
    column_name = f"D{i}"
    index2 = np.exp(C['delta1'].iloc[i] - C['delta1'].iloc[i-1])
    C.loc[0,'index2'] = 1
    C.loc[i, 'index2'] = index2
    print(column_name)
    print(index2)

#index3:
C['index3'] = np.exp(C['delta2'])
print(C.index3)

#図の作成
import matplotlib.pyplot as plt
C['x'] = range(1,25)

plt.plot(C.x, C.index1, label='INDEX_1')
plt.plot(C.x, C.index2, label='INDEX_2')
plt.plot(C.x, C.index3, label='INDEX_3')

plt.xticks(range(1, 25))
plt.legend()
plt.grid(which='major', axis='y') #grid

plt.xlabel('TIME')
plt.ylabel('INDEXES')
plt.title('Chart1:Quantity Adjusted Price Indexs')

plt.show()

##6.2 characterstics price index method
##5-2.CSVというファイルを作りました。各回帰のbeta(beta0,beta1-beta8)と各時間帯の特徴量の平均値(z1-z8)がある。
#data process
D = pd.read_csv('C:/Users/ZHOUHANGXU/Desktop/python/5-2.CSV', encoding='utf-8-sig')
D.columns = [col.replace('\ufeff', '') for col in D.columns]

#laseyre index
D['indexl'] = 1

for i in range(1,25):
    column_name = f"D{i}"  
    a1 = np.exp(D['beta0'].iloc[i]+D['beta1'].iloc[i]*D['z1'].iloc[i-1]+D['beta2'].iloc[i]*D['z2'].iloc[i-1]
                +D['beta3'].iloc[i]*D['z3'].iloc[i-1]+D['beta4'].iloc[i]*D['z4'].iloc[i-1]+D['beta5'].iloc[i]*D['z5'].iloc[i-1]
                +D['beta6'].iloc[i]*D['z6'].iloc[i-1]+D['beta7'].iloc[i]*D['z7'].iloc[i-1]+D['beta8'].iloc[i]*D['z8'].iloc[i-1])
    
    a2 = np.exp(D['beta0'].iloc[i-1]+D['beta1'].iloc[i-1]*D['z1'].iloc[i-1]+D['beta2'].iloc[i-1]*D['z2'].iloc[i-1]
                +D['beta3'].iloc[i-1]*D['z3'].iloc[i-1]+D['beta4'].iloc[i-1]*D['z4'].iloc[i-1]+D['beta5'].iloc[i]*D['z5'].iloc[i-1]
                +D['beta6'].iloc[i-1]*D['z6'].iloc[i-1]+D['beta7'].iloc[i-1]*D['z7'].iloc[i-1]+D['beta8'].iloc[i-1]*D['z8'].iloc[i-1])
    indexl = a1 / a2
    D.loc[i, 'indexl'] = indexl
    print(column_name)
    print(indexl)

#paasche index
D['indexp'] = 1

for i in range(1,25):
    column_name = f"D{i}"  
    a1 = np.exp(D['beta0'].iloc[i]+D['beta1'].iloc[i]*D['z1'].iloc[i]+D['beta2'].iloc[i]*D['z2'].iloc[i]
                +D['beta3'].iloc[i]*D['z3'].iloc[i]+D['beta4'].iloc[i]*D['z4'].iloc[i]+D['beta5'].iloc[i]*D['z5'].iloc[i]
                +D['beta6'].iloc[i]*D['z6'].iloc[i]+D['beta7'].iloc[i]*D['z7'].iloc[i]+D['beta8'].iloc[i]*D['z8'].iloc[i])
    a2 = np.exp(D['beta0'].iloc[i-1]+D['beta1'].iloc[i-1]*D['z1'].iloc[i]+D['beta2'].iloc[i-1]*D['z2'].iloc[i]
                +D['beta3'].iloc[i-1]*D['z3'].iloc[i]+D['beta4'].iloc[i-1]*D['z4'].iloc[i]+D['beta5'].iloc[i-1]*D['z5'].iloc[i]
                +D['beta6'].iloc[i-1]*D['z6'].iloc[i]+D['beta7'].iloc[i-1]*D['z7'].iloc[i]+D['beta8'].iloc[i-1]*D['z8'].iloc[i])
    indexp = a1 / a2
    D.loc[i, 'indexp'] = indexp
    print(column_name)
    print(indexp)

#fisher index
D['indexf'] = (D['indexl']*D['indexp'])**0.5

#図の作成
#D1 = D.drop(24)
import matplotlib.pyplot as plt
D['x'] = range(1,25)

plt.plot(D.x, D.indexl, label='INDEX_L')
plt.plot(D.x, D.indexp, label='INDEX_P')
plt.plot(D.x, D.indexf, label='INDEX_F')

plt.xticks(range(1, 25))
plt.legend()
plt.grid(which='major', axis='y') #grid

plt.xlabel('TIME')
plt.ylabel('INDEXES')
plt.title('Chart2:Quantity Adjusted Price Indexs')

plt.show()
