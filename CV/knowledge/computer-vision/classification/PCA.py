from sklearn.decomposition import PCA
N_dim =  100# 49152(=128×128×3)の列を100列に落とし込む

pca = PCA(n_components=N_dim, random_state=0)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))