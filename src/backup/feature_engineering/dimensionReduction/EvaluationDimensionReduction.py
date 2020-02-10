#降维效果的评估

#1、PCA，指标：累计方差贡献率
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.explained_variance_ratio_


#2、线性判别分析，监督学习，原特征下的分类准确率和降维后的特征下的准确率比较 来评估LDA的效果

#