#_author: Hyy
#date: 2020-10-23
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense,Lambda
from keras.utils.np_utils import to_categorical
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from scipy.io import loadmat
import argparse
from scipy.stats import norm
import pandas as pd
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from keras.losses import mse, binary_crossentropy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from Evaluate_function import Evaluate_Fun
from Relie import RReliefF,ReliefF,Relief
import numpy as np
import h5py
import time

n_classes = 2


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# data_file = h5py.File('H:/save/data/Colon/g.mat','r')
# # data_file = ('H:/save/data/Colon/g.mat')
# label_path = h5py.File('H:/save/data/Colon/h.mat','r')
# # label_struct = ('H:/save/data/Colon/h.mat')
# # data = loadmat(data_file, mat_dtype=True)
# # label = loadmat(label_struct, mat_dtype=True)
# X = data_file['g']
# # X = X.T
# y = label_path['h']
# X = scale(X)
# # print(np.isinf(X).any(),np.isfinite(X).all(),np.isnan(X).any())
# # print(np.nonzero(pd.isnull(X)))
#
# X = np.transpose(X)
# #
# # # X=np.array(X)
# # # y=np.array(y)
# # #
# y=np.transpose(y)
# y=np.squeeze(y)
# # print(np.unique(y))
# print(y.shape,X.shape)
# # # # # print(np.unique(y))
#
# data = pd.read_csv('H:/hyy/data/TCGA/a-smotch/data/sample/gene&cnv_1000.csv',
# 					   index_col=0, header=None, lineterminator="\n", error_bad_lines=False, encoding="utf-8")
# data = data.values
# 	# out_file = 'H:/hyy/data/TCGA/a-smotch/data/sample name/var_imp_gene'
# 	# expression = np.loadtxt(file, dtype=float, delimiter="", skiprows=1)
#
# data_gene1 = data[:, 0:372]
# data_gene2 = data[:, 372:402]
# data_gene = np.hstack((data_gene1, data_gene2))
#
# label = data_gene[0, :]
# gene_data = data_gene[1:, :]
#
# gene_data = gene_data.T
# Y = np.array(label.astype(int))
# min_max_scaler = preprocessing.MinMaxScaler()
# x1 = min_max_scaler.fit_transform(gene_data)
file = ('./data/Lymphoma/y1.mat')
label_struct = ('./data/Lymphoma/z1.mat')
	# mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
data = loadmat(file, mat_dtype=True)
label = loadmat(label_struct, mat_dtype=True)
	# 导入后的data是一个字典，取出想要的变量字段即可。
X = data['y']
# X = X.T
print(X.shape)
y = label['z'].reshape(-1)
print(y.shape)
# print(np.unique(y))
sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))  # 表示剔除特征的方差大于阈值的特征Removing features with low variance
X_new = sel.fit_transform(X)
print(X_new.shape)


# # n_classes = 2
# #
feature_list = dict()
W = RReliefF(X_new, y)
W = W.reshape(-1)
#
for id, w in enumerate(W):
    if w > 0.01:
        feature_list.update({id: w})
feature_list = list(feature_list.keys())
print(X_new[:, feature_list].shape)

X_new = X_new[:, feature_list]

n_trees = 1000
rfc = RandomForestClassifier(n_estimators=n_trees,n_jobs=-1,bootstrap=True,oob_score=True)
# clf = rfc.fit()
clf = rfc.fit(X_new, y)#,max_depth=20
# print(clf.oob_score_)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_new)
print(X_new.shape)

# # importances = clf.feature_importances_
# # std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
# # indices = np.argsort(importances)[::-1]
# # # Print the feature ranking
# # importanceindex = []
# # print("Feature ranking:")
# # for f in range(X_new.shape[1]):
# #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#
#
x_train, x_test,y_train, y_test= train_test_split(X_new,y,test_size=0.2,random_state=42)
# Y_train = to_categorical(y_train, n_classes)
# Y_test = to_categorical(y_test, n_classes)

# plt.figure(figsize=(6,6))
# plt.scatter(X_new[:,0],X_new[:,1],c=y_test)
# plt.show()

hidden_1 = 256
hidden_2 = 128
hidden_3 = 64
layer_out = 2

#编码层
input_layer = Input(shape=(x_train.shape[1],))
encoded_h1 = Dense(hidden_1, activation='relu')(input_layer)
# encoded_h2 = Dense(hidden_2,activation='relu')(encoded_h1)
# encoded_h3 = Dense(hidden_3,activation='relu')(encoded_h2)

z_mean = Dense(layer_out)(encoded_h1)
z_log_var = Dense(layer_out)(encoded_h1)

z = Lambda(sampling, output_shape=(layer_out,))([z_mean, z_log_var])

encoder = Model(input_layer, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# 解码层
# decoder_h = Dense(hidden_1, activation='relu')
# decoder_mean = Dense(x_train.shape[1], activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)
#
# vae = Model(input_layer, x_decoded_mean)
# #
# parser = argparse.ArgumentParser()
# help_ = "Load h5 model trained weights"
# parser.add_argument("-w", "--weights", help=help_)
# help_ = "Use mse loss instead of binary cross entropy (default)"
# parser.add_argument("-m",
#                         "--mse",
#                         help=help_, action='store_true')
# args = parser.parse_args()
# if args.mse:
#     xent_loss = mse(input_layer, x_decoded_mean)
# else:
#     xent_loss = binary_crossentropy(input_layer,
#                                               x_decoded_mean)
#
# # xent_loss = K.mean(MSE(input_layer,outputs), axis=1)
# # xent_loss = mse(input_layer, x_decoded_mean)
# kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
# vae_loss = K.mean(xent_loss + kl_loss)
#
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')
# vae.summary()
# plot_model(vae,
#                to_file='vae_mlp.jpg',
#                show_shapes=True)
#
# epochs = 50
# batch_size = 25
# if args.weights:
#     vae = vae.load_weights(args.weights)
# else:
#     # train the autoencoder
#     vae.fit(x_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_data=(x_test, None))

#
# print('test accuracy')
# vae_features = vae.predict(x_train)

# clf = SVC(C=1, kernel='rbf').fit(vae_features, y_train)
# vae_features_test = vae.predict(x_test)
# pred = clf.predict(vae_features_test)
# Evaluate_Fun(pred, y_test, x_test)
print('=============================start encoder predict============================================')

encoder = Model(input_layer, z_mean)
encoder.summary()

print('test accuracy')
encoder_features = encoder.predict(x_train)
print(encoder_features.shape)
encoder_features_test = encoder.predict(x_test)

start_time = time.time()


clf = SVC(C=1, kernel='rbf').fit(encoder_features, y_train)
pred = clf.predict(encoder_features_test)
acc = accuracy_score(y_test,pred)
print('acc = %f' %acc)
# Evaluate_Fun(pred, y_test, x_test)

print("Total time used: %s seconds " % (time.time() - start_time) )
print('========================================end encoder predict===================================================================================')
# print("train accuracy")
# print('test accuracy')
# clf = SVC(C = 1,kernel='rbf').fit(x_train,y_train)
#
# pred = clf.predict(x_test)
# Evaluate_Fun(pred,y_test,x_test)

#
# knn = KNeighborsClassifier().fit(encoder_features,y_train)
# pred = knn.predict(encoder_features_test)
# # print(pred)
# Evaluate_Fun(pred,y_test,x_test)
# score = clf.score(encoder_features,y_test)

# bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
#                    algorithm="SAMME",
#                    n_estimators=200, learning_rate=0.8)
# bdt.fit(encoder_features, y_train)
# pred = bdt.predict(encoder_features_test)
# print(pred)
# Evaluate_Fun(pred, y_test, x_test)
