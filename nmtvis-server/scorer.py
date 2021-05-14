import math


class Scorer:
    def compute_scores(self, source, translation, attention, keyphrases, reference):
        return {"coverage_penalty": self.coverage_penalty(attention),
                "coverage_deviation_penalty": self.coverage_deviation_penalty(attention),
                "confidence": self.confidence(attention),
                "length": len(source.replace("@@ ", "").split(" ")),
                "ap_in": self.absentmindedness_penalty_in(attention),
                "ap_out": self.absentmindedness_penalty_out(attention),
                "keyphrase_score": self.keyphrase_score(source, keyphrases, attention),
                "similarityToSelectedSentence": self.similarityToSelectedSentence(source, reference),
                "length_deviation": self.length_deviation(source, translation)}


    def similarityToSelectedSentence(self, sentence, reference):
        score = 0
        if reference == "" or sentence == "":
            return score

        # https://www.geeksforgeeks.org/python-measure-similarity-between-two-sentences-using-cosine-similarity/

        # Program to measure similarity between
        # two sentences using cosine similarity.
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        X = sentence.replace("@@ ", "")
        Y = reference.replace("@@ ", "")

        # tokenization
        X_list = word_tokenize(X)
        Y_list = word_tokenize(Y)

        # sw contains the list of stopwords
        sw = stopwords.words('english')
        l1 = []
        l2 = []

        # remove stop words from string
        X_set = {w for w in X_list if not w in sw}
        Y_set = {w for w in Y_list if not w in sw}

        # form a set containing keywords of both strings
        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0

        # cosine formula
        for i in range(len(rvector)):
            c += l1[i] * l2[i]
        cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
        #print("similarity: ", cosine)

        score = cosine

        return score


    def keyphrase_score(self, sentence, keyphrases, attention):
        score = 0

        for word in sentence.replace("@@ ", "").split(" "):
            for keyphrase, freq in keyphrases:
                score += word.lower().count(keyphrase.lower()) * freq
        return score


    def length_deviation(self, source, translation):
        source = source.split(" ")
        translation = translation.split(" ")

        X, Y = len(source), len(translation)

        return math.fabs(X - Y) / X


    def coverage_penalty(self, attention, beta=0.4):
        X = len(attention[0])
        Y = len(attention)

        res = 0
        for i in range(X):
            sum_ = 0
            for j in range(Y-1):
                sum_ += attention[j][i]
            res += math.log(min(1, sum_)) if sum_ > 0 else 0

        return -beta * res


    def coverage_deviation_penalty(self, attention):
        m, n = len(attention), len(attention[0])

        res = 0
        for j in range(n):
            res += math.log(1 + (1 - sum([attention[i][j] for i in range(m-1)])) ** 2)
        return (1 / n) * res


    def absentmindedness_penalty_out(self, attention):
        m, n = len(attention), len(attention[0])

        sum_ = 0
        for row in attention[:-1]:
            norm = sum(row)
            if norm > 0:
                normRow = [i / norm for i in row]
                sum_ += sum([(i * math.log(i) if i else 0) for i in normRow])

        return - (1 / m) * sum_


    def absentmindedness_penalty_in(self, attention):
        return self.absentmindedness_penalty_out(attention)


    def confidence(self, attention):
        x = self.coverage_deviation_penalty(attention) + self.absentmindedness_penalty_in(
            attention) + self.absentmindedness_penalty_out(attention)

        return math.exp(-0.05 * (x ** 2))


    def length_penalty(self, attention):
        return len(attention[0])
