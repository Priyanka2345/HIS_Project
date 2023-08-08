
import pandas as pd
import re
import nltk
import time
import matplotlib.pyplot as plt
from PIL import Image
from pyLDAvis import display
import pyLDAvis.gensim_models as gensimvis
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from wordcloud import WordCloud
from matplotlib import colors as mcolors
from read_data import read_data


# Preprocess text data
# Define the regular expression patterns
pattern1 = r'^(c-\d+-\d+)\s+X\s+(.*)$'
pattern2 = r'^(readme-\d+)\s+X\s+(.*)$'
df1, df2 = read_data(pattern1, pattern2)


def preprocess_text(text):
    result = []
    stemmer = SnowballStemmer(language='english')
    lemmatizer = WordNetLemmatizer()

    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            stemmed_token = stemmer.stem(lemmatizer.lemmatize(token, pos='v'))
            result.append(stemmed_token)

    return result

# Read and preprocess data


def preprocess_data(df, text_column):
    documents = df[text_column].map(preprocess_text)
    return documents

# Visualize topic modeling


def visualize_topics(lda_model, corpus, dictionary):
    lda_display = gensimvis.prepare(
        lda_model, corpus, dictionary, sort_topics=False)
    display(lda_display)

# Build LDA model


def build_lda_model(documents, num_topics):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    lda_model = models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha='auto', eta='auto')
    return lda_model, corpus, dictionary


# Output A) Parameters, B) Parameters, and N) Parameters
def output_model_parameters(lda_model):
    alpha = lda_model.alpha[0]
    beta = lda_model.eta[0]
    num_topics = lda_model.num_topics

    time.sleep(1)

    return alpha, beta, num_topics


# Wordcloud of Top N words in each topic


def visualize_wordcloud(lda_model):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    stop_words = STOPWORDS

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.ion()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('plot.png')
    open_plot_image('plot.png')


def open_plot_image(filename):
    img = Image.open(filename)
    img.show()
