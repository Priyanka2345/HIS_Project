from read_data import *
from model_ import *
import time
nltk.download('wordnet')

start = time.time()


def main():
    # Preprocess data
    documents1 = preprocess_data(df1, 'Value')
    documents2 = preprocess_data(df2, 'Value')

    # Merge documents
    documents = pd.concat([documents1, documents2], ignore_index=True)

    # Build LDA model
    num_topics = 20
    lda_model, corpus, dictionary = build_lda_model(documents, num_topics)

    # Visualize topic modeling results
    visualize_topics(lda_model, corpus, dictionary)

    # Output model parameters
    A, B, N = output_model_parameters(lda_model)

    print("Alpha value:", A)
    print("Beta value:", B)
    print("Num topics value:", N)

    # Visualize word cloud
    visualize_wordcloud(lda_model)


if __name__ == '__main__':

    main()
    end = time.time()
    print("Running time: %s Seconds" % (end - start))
