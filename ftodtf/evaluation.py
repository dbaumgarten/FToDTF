""" Handles all evaluation relatet tasks, like printing the nearest neighbors of a word or visualizing the generated vectors"""
import os
from tempfile import gettempdir
from tensorflow.contrib.tensorboard.plugins import projector

def printSimilarityCheck(similarityop,valid_examples,valid_size,reverse_dictionary):
    sim = similarityop.eval()
    for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

def tensorboardVisualisation(embeddings,writer,log_dir):
  config = projector.ProjectorConfig()
  # pylint: disable=no-member
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

def visualizeEmbeddings(reverse_dictionary,final_embeddings,dir):
  try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')
        plt.savefig(filename)

    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(dir, 'tsne.png'))

  except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)