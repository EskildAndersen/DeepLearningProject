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