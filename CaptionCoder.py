# script der skal start token,end token og pad sætningen til en maks længde
# input er En sætning, et vocab og en maks længde1

from vocabulary import inv_vocab, vocab, max_len, example


def tokenizeCaptions(sentence, vocab, maxLength):

    SOS = vocab.get("<SOS>")
    EOS = vocab.get("<EOS>")
    PAD = vocab.get("<PAD>")

    outputSentence = [vocab.get(word) for word in sentence.split()]
    outputSentence = [SOS] + outputSentence + [EOS] + \
        [PAD]*(maxLength - len(outputSentence)-2)
    return outputSentence


def deTokenizeCaptions(tokenizedSentence, invVocab, asString=False):

    outputSentence = [invVocab.get(word) for word in tokenizedSentence]

    if asString:
        removeTokens = ["<SOS>", "<EOS>", "<PAD>"]
        strOut = " "
        outputSentence = [
            word for word in outputSentence if word not in removeTokens]
        outputSentence = strOut.join(outputSentence)
    return outputSentence


# token = tokenizeCaptions(example,vocab,max_len)

# deTokenList = deTokenizeCaptions(token,inv_vocab)

# deTokenWord = deTokenizeCaptions(token,inv_vocab,True)

# print(token)
# print(deTokenList)
# print(deTokenWord)
