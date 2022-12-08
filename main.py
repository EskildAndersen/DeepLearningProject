'''
Main script to run training, validation and testing of the model given settings defined in settings.py
Script also plots relevant metrics such as loss.

'''


from train import train_main
import HelperFunctions as hf

def main():
    # Save settings
    settings = hf.saveSettings()
    
    # Run Training loop (include validation)
    train_main(settings)
    
    # Do some plotting
    

if __name__ == '__main__':
    main()