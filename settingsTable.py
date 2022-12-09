import os
import pandas as pd

def createDataFrame():
    '''
    Creates a table overview of all settings and their best accuracies
    '''
    path_settings = os.path.join('results','settings')
    results_settings = os.path.join('results','evaluation')

    finaldf = pd.DataFrame()

    for file in os.listdir(path_settings)[1:][::-1]:
        setting = file.split('.')[0]
        df = pd.read_csv(os.path.join(path_settings,file),sep=':',header=None)
        df.columns = ['parameter',setting]
        df.set_index('parameter',inplace=True,drop=True)
        df = df.transpose()
        test_acc = pd.read_csv(os.path.join(results_settings,f'{setting}_train_acc.txt'),sep=',')
        test_acc.columns = test_acc.columns.str.replace(" ","")
        Bleu1Test = test_acc['BLEU1'].round(2).max()
        Bleu2Test = test_acc['BLEU2'].round(2).max()
        Bleu3Test = test_acc['BLEU3'].round(2).max()
        Bleu4Test = test_acc['BLEU4'].round(2).max()
        df['Bleu1 Test'] = Bleu1Test
        df['Bleu2 Test'] = Bleu2Test
        df['Bleu3 Test'] = Bleu3Test
        df['Bleu4 Test'] = Bleu4Test
        dev_acc = pd.read_csv(os.path.join(results_settings,f'{setting}_dev_acc.txt'),sep=',')
        dev_acc.columns = dev_acc.columns.str.replace(" ","")
        Bleu1Dev = dev_acc['BLEU1'].round(2).max()
        Bleu2Dev = dev_acc['BLEU2'].round(2).max()
        Bleu3Dev = dev_acc['BLEU3'].round(2).max()
        Bleu4Dev = dev_acc['BLEU4'].round(2).max()
        df['Bleu1 Dev'] = Bleu1Dev
        df['Bleu2 Dev'] = Bleu2Dev
        df['Bleu3 Dev'] = Bleu3Dev
        df['Bleu4 Dev'] = Bleu4Dev
        finaldf = finaldf.append(df)
        pass


    finaldf.columns = finaldf.columns.str.replace(" ","")
    finaldf['LOSS_PAD_INDEX'] = finaldf['LOSS_PAD_INDEX'].apply(lambda x: None if x == ' -100' else x)

    finaldf = finaldf.transpose()
    return finaldf



if __name__ == '__main__':
    df = createDataFrame()
    df.to_excel('table.xlsx')
