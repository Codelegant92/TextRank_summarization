#!/usr/bin/env python
# coding=utf-8

import numpy as np
import jieba

class textRank(object):
    def __init__(self, read_filePath, output_filePath):
        self.read_filePath = read_filePath
        self.output_filePath = output_filePath

    def _read(self):
        preprocessedFile = 'processedFile.txt'
        endPunc = ['。', '！', '?']
        document = {}
        with open(self.read_filePath, 'r') as f:
            s = f.read()
            f.close()
        seglist1 = jieba.cut(s, cut_all=False)
        sentence = []
        i = 0
        seglist = []
        for seg in seglist1:
            if seg != '\n':
                seglist.append(seg)
        with open(preprocessedFile, 'w') as f:
            for item in seglist:
                uitem = item.encode('utf8')
                if uitem not in endPunc:
                    sentence.append(uitem)
                else:
                    writeSent = str(i) + ' ' + ' '.join(sentence).strip() + '\n'
                    document[i] = writeSent
                    f.write(writeSent)
                    sentence = []
                    i += 1
            f.close()
        return document, preprocessedFile
        # document = 0
        # processedFile = 1

    def _generateVector(self, preprocessedFile):
        document = []
        docDict = {}
        newDocument = []
        with open(preprocessedFile, 'r') as f:
            for line in f.readlines():
                document.append(line.split(' ')[1:])
            f.close()
        i = 0
        for sentence in document:
            newSentence = []
            for word in sentence:
                if word not in docDict.keys():
                    docDict[word] = i
                    i += 1
                newSentence.append(docDict[word])
            newDocument.append(newSentence)
        X = np.zeros([len(document), len(docDict)])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i][j] = newDocument[i].count(j)
        print 'generated vector X: %d %d' % (X.shape[0], X.shape[1])
        return (X)

    def sentValue(self):
        '''
            This function is to iterally calculate the values of each sentence of a news article.

            Input: X. (N, D), Each row represents a sentence. Each column represents a term in the dictionary.
            X[i][j] represents the term frequency of term i in sentence j. N is the number of sentences in the
            article, D is the size of the dictionary.
            Output: S. (N, ).
            '''
        # Initialize
        doc, preprocessedFile = self._read()
        X = self._generateVector(preprocessedFile)
        N, D = X.shape
        W = np.zeros([N, N])
        S = np.ones(N) * 1.0 / N
        C = np.zeros([N, N])
        iter_num = 100

        # calculate matrix W
        for i in range(N):
            for j in range(N):
                # Euclid(2-norm)
                # W[i][j] = np.sum((X[i] - X[j]) ** 2)*1.0
                # City(1-norm)
                # W[i][j] = np.sum(np.abs(X[i] - X[j]))
                if i != j:
                    common_term = (X[i] * X[j] > 0)
                    if np.sum(X[i] * common_term + X[j] * common_term) > 0:
                        C[i][j] = np.sum(X[i] * common_term) * 1.0 / np.sum(X[i] * common_term + X[j] * common_term)
        # Cos
        W = np.dot(X, X.T) / np.dot(np.sqrt(np.sum(X ** 2, axis=1)).reshape([1, N]),
                                    np.sqrt(np.sum(X ** 2, axis=1)).reshape([N, 1]))

        # calculate c
        cc = np.sum(C * W, axis=1)
        delta_w = np.ones([N, N]) * (W != 0) * (1 - np.eye(N)) * 1.0
        #print 'cc:'
        #print(cc)
        #print 'delta_w:'
        #print(delta_w)
        # iterate S
        for iter in range(iter_num):
            print 'iter %d' % (iter)
            print(S)
            oldS = S
            S = (S + cc) / (S + cc + np.dot(S + cc, delta_w.T))
            if (np.sum(np.abs(S - oldS) / oldS) < 1e-7):
                #print(np.abs(S - oldS) / oldS)
                print 'stop here at iter %d!' % (iter)
                idx = [i for i, s in sorted(enumerate(S), key=lambda x: -x[-1])]
                rank = [(i, s) for i, s in enumerate(S)]
                with open(self.output_filePath, 'w') as f:
                    for item in idx:
                        f.write(doc[item] + '\n')
                    f.close()

                return rank
        return 0

    
if __name__ == "__main__":
    sent = textRank('input.txt', 'output.txt')
    out = sent.sentValue()
    print out


