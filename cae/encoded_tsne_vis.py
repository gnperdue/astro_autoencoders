import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.manifold import TSNE

from tfcae.data_readers import read_encoded_hdf5


nfloats = 64
prefix = 'stargalaxy_sim_20190214'

datafile = prefix + 'encoded_test.hdf5'
images, labels = read_encoded_hdf5(datafile, nfloats)

feat_cols = ['element' + str(i) for i in range(images.shape[1])]
df = pd.DataFrame(images, columns=feat_cols)
df['labels'] = labels
# df['labels'] = df['labels'].apply(lambda i: str(i))

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.gray()
fig = plt.figure(figsize=(16, 7))
for i in range(0, 15):
    ax = fig.add_subplot(3, 5, i + 1,
                         title='{}'.format(str(df.loc[rndperm[i], 'labels'])))
    ax.matshow(
        df.loc[rndperm[i], feat_cols].values.reshape((8, 8)).astype(float)
    )
    ax.axis('off')
# plt.show()
plt.savefig('cae_encodeds.pdf', bbox_inches='tight')

N = 10000
df_subset = df.loc[rndperm[:N], :].copy()
data_subset = df_subset[feat_cols].values

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="labels",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)
# plt.show()
plt.axis('off')
plt.savefig('cae_tsne.pdf', bbox_inches='tight')
