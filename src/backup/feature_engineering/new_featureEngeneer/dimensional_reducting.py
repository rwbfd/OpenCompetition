import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold
from sklearn.lda import LDA

def dimensionalReduction(df_list,method,methods_list):
    '''
    
    :param df_list: df
    :param methods_list: methods of dimension reduction
    :return: 
    '''
    #methods = {'pca':[n_components],'lda':[n_components],'tSNE':[n_components]}
    df_list = pd.DataFrame(df_list)
    print(df_list.shape)
    if method not in methods_list.keys():
        raise ValueError("please use method in method_list")
    if method == 'pca':
        args = methods_list[method]
        model = PCA(n_components=args)
        x_pca = model.fit_transform(df_list)
        return x_pca

    if method == 'lda':
        args = methods_list[method]
        model = LDA(n_components=args)
        y = df_list[['target']]
        x_lda = model.fit_transform(df_list.drop('target',axis=1),y)
        return x_lda

    if method == 'tSNE':
        args = methods_list[method]
        tsne = manifold.TSNE(n_components=args)
        X_tsne = tsne.fit_transform(df_list)
        return X_tsne

    if method == 'lle':
        args = methods_list[method]
        lle = manifold.LocallyLinearEmbedding(n_components=args)
        x_lle = lle.fit_transform(df_list)
        return x_lle

    if method == 'isomap':
        args = methods_list[method]
        x_iso = manifold.Isomap(n_components=args).fit_transform(df_list)
        return x_iso

    if method == 'mds':
        args = methods_list[method]
        x_mds = manifold.MDS(n_components=args).fit_transform(df_list)
        return x_mds

