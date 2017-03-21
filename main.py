# import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# prepare corpus
corpus = []
for d in range(1400):
    f = open("./d/" + str(d + 1) + ".txt")
    corpus.append(f.read())

# add query to corpus
for q in range(225):
    f = open("./q/" + str(q + 1) + ".txt")
    corpus.append(f.read())

relevant = []
for r in range(225):
    f = open("./r/" + str(r + 1) + ".txt")
    relevant.append([int(line.rstrip('\n')) for line in f])

# init vectorizer
binary_vectorizer = TfidfVectorizer(use_idf=False, norm=False, binary=True)
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer()

# prepare matrix
binary_matrix = binary_vectorizer.fit_transform(corpus)
tf_matrix = tf_vectorizer.fit_transform(corpus)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)


fileQuality = open("quality.csv", 'w')

fileQuality.write("query,"
        "binary euclidean prec,binary euclidean rec,binary euclidean f-measure,"
        "binary cosine prec,binary cosine rec,binary cosine f-measure,"
        "tf euclidean prec,tf euclidean rec,tf euclidean f-measure,"
        "tf cosine prec,tf cosine rec,tf cosine f-measure,"
        "tfidf euclidean prec,tfidf euclidean rec,tfidf euclidean f-measure,"
        "tfidf cosine prec,tfidf cosine rec,tfidf cosine f-measure\n")

file1 = open("relevance-binary-euclidean.csv", 'a')
file2 = open("relevance-binary-cosine.csv", 'a')
file3 = open("relevance-tf-euclidean.csv", 'a')
file4 = open("relevance-tf-cosine.csv", 'a')
file5 = open("relevance-tfidf-euclidean.csv", 'a')
file6 = open("relevance-tfidf-cosine.csv", 'a')


for q in range(225):
    # compute similarity between query and all docs (tf-idf) and get top 10 relevant
    y_pred = [True, True, True, True, True, True, True, True, True, True]

    sim = np.array(euclidean_distances(binary_matrix[1400 + q], binary_matrix[0:1400])[0])
    for s in range(1399):
        file1.write(`sim[s]` + ",")
    file1.write(`sim[1399]` + "\n")

    topRelevant = sim.argsort()[-10:][::-1] + 1
    y_true = []
    for t in range(10):
        if topRelevant[t] in relevant[q]:
            y_true.append(True)
        else:
            y_true.append(False)
    res1 = precision_recall_fscore_support(y_true, y_pred, average='binary')


    sim = np.array(cosine_similarity(binary_matrix[1400 + q], binary_matrix[0:1400])[0])
    for s in range(1399):
        file2.write(`sim[s]` + ",")
    file2.write(`sim[1399]` + "\n")

    topRelevant = sim.argsort()[-10:][::-1] + 1
    y_true = []
    for t in range(10):
        if topRelevant[t] in relevant[q]:
            y_true.append(True)
        else:
            y_true.append(False)
    res2 = precision_recall_fscore_support(y_true, y_pred, average='binary')


    sim = np.array(euclidean_distances(tf_matrix[1400 + q], tf_matrix[0:1400])[0])
    for s in range(1399):
        file3.write(`sim[s]` + ",")
    file3.write(`sim[1399]` + "\n")

    topRelevant = sim.argsort()[-10:][::-1] + 1
    y_true = []
    for t in range(10):
        if topRelevant[t] in relevant[q]:
            y_true.append(True)
        else:
            y_true.append(False)
    res3 = precision_recall_fscore_support(y_true, y_pred, average='binary')


    sim = np.array(cosine_similarity(tf_matrix[1400 + q], tf_matrix[0:1400])[0])
    for s in range(1399):
        file4.write(`sim[s]` + ",")
    file4.write(`sim[1399]` + "\n")

    topRelevant = sim.argsort()[-10:][::-1] + 1
    y_true = []
    for t in range(10):
        if topRelevant[t] in relevant[q]:
            y_true.append(True)
        else:
            y_true.append(False)
    res4 = precision_recall_fscore_support(y_true, y_pred, average='binary')


    sim = np.array(euclidean_distances(tfidf_matrix[1400 + q], tfidf_matrix[0:1400])[0])
    for s in range(1399):
        file5.write(`sim[s]` + ",")
    file5.write(`sim[1399]` + "\n")

    topRelevant = sim.argsort()[-10:][::-1] + 1
    y_true = []
    for t in range(10):
        if topRelevant[t] in relevant[q]:
            y_true.append(True)
        else:
            y_true.append(False)
    res5 = precision_recall_fscore_support(y_true, y_pred, average='binary')


    sim = np.array(cosine_similarity(tfidf_matrix[1400 + q], tfidf_matrix[0:1400])[0])
    for s in range(1399):
        file6.write(`sim[s]` + ",")
    file6.write(`sim[1399]` + "\n")

    topRelevant = sim.argsort()[-10:][::-1] + 1
    y_true = []
    for t in range(10):
        if topRelevant[t] in relevant[q]:
            y_true.append(True)
        else:
            y_true.append(False)
    res6 = precision_recall_fscore_support(y_true, y_pred, average='binary')

    fileQuality.write(`q` + "," +
          `res1[0]` + "," + `res1[1]` + "," + `res1[2]` + "," +
          `res2[0]` + "," + `res2[1]` + "," + `res2[2]` + "," +
          `res3[0]` + "," + `res3[1]` + "," + `res3[2]` + "," +
          `res4[0]` + "," + `res4[1]` + "," + `res4[2]` + "," +
          `res5[0]` + "," + `res5[1]` + "," + `res5[2]` + "," +
          `res6[0]` + "," + `res6[1]` + "," + `res6[2]` + "\n")