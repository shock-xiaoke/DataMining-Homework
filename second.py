import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


# 对points属性进行离散化并加上前缀
def points_discretization(value):
    return "points_class:" + str(int(value / 5))


# 将读取到的dataframe转换为列表
def deal(data):
    return data.to_list()


data_frame_wine13 = pd.read_csv("F:/dataMining/winemag-data-130k-v2.csv", index_col=[0])
print(data_frame_wine13.head(2))

# # 打印数据的列信息，并判断是否为数值属性,删除不使用的属性
data_frame_wine13 = data_frame_wine13.drop(
    ['designation','description','price','province','region_1','region_2','taster_twitter_handle','title','variety','winery']
    , axis=1)
print(data_frame_wine13.info())
print(data_frame_wine13.head(2))

def drop_Nan(df):
    df_new = df.copy(deep=True)
    return df_new.dropna(axis=0)

# 寻找country属性为空的数据
data_frame_wine13 = drop_Nan(data_frame_wine13)
data_frame_wine13.loc[:, 'points'] = data_frame_wine13['points'].map(lambda x: points_discretization(x))

data_frame_wine13_arr = data_frame_wine13.apply(deal, axis=1).tolist()
te = TransactionEncoder()
tf = te.fit_transform(data_frame_wine13_arr)
new_df = pd.DataFrame(tf, columns=te.columns_)
df_after_apriori = apriori(new_df, min_support=0.05, use_colnames=True, max_len=4).sort_values(by='support', ascending=False)
print(df_after_apriori)
print(df_after_apriori.shape)
print("------------------------------")

the_rules = association_rules(df_after_apriori, metric='confidence', min_threshold=0.45)
the_rules = the_rules.drop(['leverage', 'conviction'], axis=1)
print(the_rules.shape)
print(the_rules)

for index, row in the_rules.iterrows():
    t1 = tuple(row['antecedents'])
    t2 = tuple(row['consequents'])
    print("%36s ====> %22s  (suupport = %f, confidence = %f )" % (t1, t2, row['support'], row['confidence']))


def allconf(num):
    return num.support / max(num['antecedent support'], num['consequent support'])
def maxconf(num):
    return max(num.support/num['antecedent support'],num.support/num['consequent support'])

allconf_list = []
maxconf_list = []
for index, row in the_rules.iterrows():
    allconf_list.append(allconf(row))
    maxconf_list.append(maxconf(row))
the_rules['allconf'] = allconf_list
the_rules['maxconf'] = maxconf_list
the_rules.drop(['antecedent support', 'consequent support'], axis=1, inplace=False)
the_rules.sort_values(by=['lift'], ascending=False)[:16]

# count = 1
# for index, row in the_rules.iterrows():
#     t1 = tuple(row['antecedents'])
#     t2 = tuple(row['consequents'])
#     print("%2d : %23s ====> %20s (suupport = %f, confidence = %f )"%(count,t1,t2,row['support'],row['confidence']))
#     count = count + 1


import  matplotlib.pyplot as plt
plt.xlabel('support')
plt.ylabel('confidence')
for i in range(the_rules.shape[0]):
    plt.scatter(the_rules.support[i],the_rules.confidence[i],c='r')
plt.xlabel('support')
plt.ylabel('lift')
for i in range(the_rules.shape[0]):
    plt.scatter(the_rules.support[i],the_rules.lift[i],c='g')