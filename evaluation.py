from settings import DEVICE, BATCH_SIZE

import torch 
from vocabulary import max_len, SOS_token
from DataPreparator import ImageDataset
from ignite.metrics.nlp import Bleu
from CaptionCoder import deTokenizeCaptions


def evaluate(   
        decoder,
        type: str,
    ):  
        decoder.eval()

        # select train, dev or test
        train_dataset = ImageDataset(f'{type}_labels.txt',False)
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # initialize Bleu
        Bleu1 = Bleu(ngram = 1)
        Bleu2 = Bleu(ngram = 2)
        Bleu3 = Bleu(ngram = 3)
        Bleu4 = Bleu(ngram = 4)

        for labels,input_sentences, input_features in dataloader:
            
            # send to device
            input_features = input_features.to(DEVICE)
            input_sentences = input_sentences.to(DEVICE)
            
            batch_size = input_features.shape[0]

            # initialize for decoder
            prediction = torch.full((batch_size,), SOS_token).to(DEVICE)
            predictions = []

            output, _ = decoder(input_sentences[:,0,:], input_features)

                # calculate prediction to use as the next input
            prediction = output.argmax(-1)
            predictions.append(prediction)
                
            # prep final prediction and labels for Bleu calculations
            predictions = torch.stack(predictions, 1)
            label = zip(*labels)

            # split sentences to list of words
            predictionssplit = [deTokenizeCaptions(prediction,asString=True).split() for prediction in predictions.squeeze(1)]
            labelsplit = [[l.split() for l in ls] for ls in label]

            # update Bleu scores
            Bleu1.update((predictionssplit,labelsplit))
            Bleu2.update((predictionssplit,labelsplit))
            Bleu3.update((predictionssplit,labelsplit))
            Bleu4.update((predictionssplit,labelsplit))
        
        # Compute the Bleu scores for the entire dataset
        bleu1score = Bleu1.compute()
        bleu2score = Bleu2.compute()
        bleu3score = Bleu3.compute()
        bleu4score = Bleu4.compute()

        return bleu1score,bleu2score,bleu3score,bleu4score



if __name__ == '__main__':

    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    encoder = torch.load('encoder_model.pt',map_location=device)
    encoder.device = device # since only one gpu quick fix

    decoder = torch.load('decoder_model.pt',map_location=device)
    decoder.device = device # since only one gpu quick fix

    bleu1,bleu2, bleu3, bleu4 = evaluate(encoder,decoder,'test')
