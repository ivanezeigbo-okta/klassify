import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
from kneed import KneeLocator
import math
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go


with open('all_bugs.json') as f:
  data = json.load(f)

issues = data['issues']
'''
with open('../../Documents/all_bugs1.json') as f:
  data1 = json.load(f)

issues += data1['issues']

with open('../../Documents/all_bugs2.json') as f:
  data2 = json.load(f)

issues += data2['issues']

with open('../../Documents/all_bugs3.json') as f:
  data3 = json.load(f)

issues += data3['issues']

with open('../../Documents/all_bugs4.json') as f:
  data4 = json.load(f)

issues += data4['issues']

with open('../../Documents/all_bugs5.json') as f:
  data5 = json.load(f)

issues += data5['issues']
'''

sentences = []

for every in issues:
    main_data = every['fields']
    all_string = str(main_data['summary']) + ' ' + str(main_data['description']) + ' ' + str(every['id'])
    for each in main_data['issuelinks']:
        a = ' ' + str(each['id'])
        all_string += a
    sentences.append(all_string)


def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing)
X = tfidf_vectorizer.fit_transform(sentences)

x_range = []

counter = 1
wcss = []
maxK_Found = False
while not maxK_Found and counter <= len(issues):
    kmeans = KMeans(n_clusters = counter, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    x_range.append(counter)
    if counter != 1:
        previous_wcss = wcss[-1]
        if kmeans.inertia_ >= previous_wcss:
            maxK_Found = True
    counter += 1
    wcss.append(kmeans.inertia_)

plt.plot(x_range, wcss)
plt.title('The Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares (WCSS)')
plt.show()

kn = KneeLocator(x_range, wcss, curve='convex', direction='decreasing')
print("")
print("There are",  kn.knee,  "clusters")
print("")

n_clusters = kn.knee
clf = KMeans(n_clusters=n_clusters, max_iter = 100, init='k-means++', n_init=1, random_state=42)
labels = clf.fit_predict(X)
#print(labels)

all_jiras = {}
for index, sentence in enumerate(sentences):
    if labels[index] not in all_jiras:
        all_jiras[labels[index]] = ["https://oktainc.atlassian.net/browse/" + str(issues[index]['key'])]
    else:
        all_jiras[labels[index]].append("https://oktainc.atlassian.net/browse/" + str(issues[index]['key']))

unique, counts = np.unique(labels, return_counts=True)
res = dict(zip(unique, counts))
res = sorted(res.items(), key=lambda x:x[1], reverse=True)

pca = SparsePCA(n_components=2).fit(X.toarray())
coords = pca.transform(X.toarray())
label_colors = ['#2AB0E9', '#2BAF74', '#D7665E', '#CCCCCC', '#D2CA0D', '#522A64', '#A3DB05', '#FC6514']
colors = [label_colors[i % len(label_colors)] for i in labels]
plt.scatter(coords[:, 0], coords[:, 1], c=colors)
centroids = clf.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.title("Principal Component Analysis Diagram of Classes")
plt.scatter(centroid_coords[:, 0], centroid_coords[:,1], marker='X', s=200, linewidth=2, c='#444d61')
plt.show()

rank_array = [str('<b>' + str(ind) + '</b>') for ind in range(1, n_clusters + 1)]
class_array = []
relatedJNum_array = []
memberJ_array = []

for m in res:
    class_array.append(m[0])
    relatedJNum_array.append(m[1])

for classes in class_array:
    final_res = ''
    all_mem = all_jiras[classes]
    for n in range(len(all_mem)):
        mem = '<a href="'+ all_mem[n] +'" target="_blank" style="text-decoration:none">'+ all_mem[n] +'</a><br>'
        final_res += mem
    memberJ_array.append(final_res)


fig = go.Figure(data=[go.Table(header=dict(values=['<b>Rank</b>', '<b>Class ID</b>', '<b>Number of Related Jiras</b>', '<b>Associated Member Jiras</b>']),
                 cells=dict(values=[rank_array, class_array, relatedJNum_array, memberJ_array], line_color='darkslategray', fill=dict(color=['paleturquoise', 'white']),
                 align=['center', 'center'], font_size=12, height=30))
                     ])

fig.update_layout(
    title={
        'text': "<b>Klassify: Jira Classes for " + str(len(rank_array)) + " Clusters</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        })

fig.show()
