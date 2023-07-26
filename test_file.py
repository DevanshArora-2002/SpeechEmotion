import argparse

import CoordVitFile
import TimnetFile
import Wav2Vec2File
import WhisperFile


def train(args):
    model_type = list(args.model.split("_"))
    if (model_type[0] == "Wav2Vec2"):
        return Wav2Vec2File.train(args.train, args.train_dataset, args.output_dir, model_type[-1], args.epochs)
    elif (model_type[0] == "Whisper"):
        return WhisperFile.train(args.train, args.train_dataset, args.output_dir, model_type[-1], args.epochs)
    elif (model_type[0] == "Timnet"):
        return TimnetFile.train(args.train, args.train_dataset, args.output_dir, model_type[-1], args.epochs)
    elif (model_type[0] == "CoordVit"):
        return CoordVitFile.train(args.train, args.train_dataset, args.output_dir, model_type[-1], args.epochs)
    else:
        raise Exception("Please provide a model with correct header")
def main():
    parser = argparse.ArgumentParser(description="This is a simple argparse script.")

    # Add arguments
    parser.add_argument("--train", type=str, default='train', help="Type of modeling")
    parser.add_argument("--epochs",type=int, default=2, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default='output', help="Directory to store the output")
    parser.add_argument("--train_dataset", type=str, default=None, help="Name of the training dataset")
    parser.add_argument("--model",type=str,default=None,help="Pre-trained model path")

    args = parser.parse_args()

    if(args.train_dataset is None):
        raise Exception("Please specify a dataset")
    if(args.train_dataset[-4]!='.csv'):
        raise Exception("Please specify a valid dataset")
    
    a=train(args)
    print(a)
    return a

if __name__=='__main__':
    main()




