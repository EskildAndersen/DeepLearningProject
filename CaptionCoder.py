# script der skal start token,end token og pad sætningen til en maks længde
# input er En sætning, et vocab og en maks længde1


sentence = "hej med dig"
vocab = {"<SOS>":0, "<EOS>": 1, "hej": 2, "med": 3, "dig": 4, "<PAD>": 5}

def tokenizeCaptions(sentence,vocab,maxLength):

    SOS = vocab.get("<SOS>")
    EOS = vocab.get("<EOS>")
    PAD = vocab.get("<PAD>")

    outputSentence = [vocab.get(word) for word in sentence.split()]
    outputSentence = [SOS] + outputSentence + [EOS] + [PAD]*(maxLength - len(outputSentence)-2)
    return outputSentence


def deTokenizeCaptions(tokenizedSentence,vocab, asString = False):


    invVocab = {v: k for k, v in vocab.items()}
    outputSentence = [invVocab.get(word) for word in tokenizedSentence]

    if asString:
        removeTokens = ["<SOS>","<EOS>","<PAD>"]
        strOut = " "
        outputSentence = [word for word in outputSentence if word not in removeTokens]
        outputSentence = strOut.join(outputSentence)
    return outputSentence


token = tokenizeCaptions(sentence,vocab,10)

deTokenList = deTokenizeCaptions(token,vocab)

deTokenWord = deTokenizeCaptions(token,vocab,True)

print(token)
print(deTokenList)
print(deTokenWord)