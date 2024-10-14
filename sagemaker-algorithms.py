#!/usr/bin/env python

algo_dictionary = {
    "Linear Learner": {
        "Description": "Fit a linear line to your training data",
        "MLType": "Regression and Classification",
        "InputFormat": "RecordIO-protobuf, CSV", 
        "InputMethod": "File or PipeMode",
        "MLTraining": "stochastic gradient descent (SGD)",
        "Regularization": "L1 and L2",
        "Validation": "Most optimal model selected",
        "Preprocessing": "normalized and shuffled",
        "Hyperparameters": "Learning rate, batch size, L1, L2 (Wd), prediction, recall",
        "TrainingHardware": "Single or multi-machine; CPU and GPU (multi-GPU does not help)" 
    },
    "XGBoost": {
        "Description": "eXtreame Gradient Boosting. A Boosted group of decision trees",
        "MLType": "Regression and Classification",
        "InputFormat": "RecordIO-protobuf, CSV, libsvm, Parquet", 
        "InputMethod": "n/a",
        "MLTraining": "gradient descent with decision trees",
        "Regularization": "L1 and L2",
        "Validation": "n/a",
        "Preprocessing": "n/a",
        "Hyperparameters": "Subsample, Eta, Gamma, Alpha(L1), Lambda(L2), eval_metric, scale_pos_weight, max_depth",
        "TrainingHardware": "Single or multi-machine; CPU and GPU; Memory bound instances, M5, P2, P3, G4dn, G5" 
    },
    "Seq2Seq": {
        "Description": "Input is a sequence of tokens, output is a sequence of tokens. Implemented with RNNs and CNNs with attention.",
        "MLType": "Regression and Classification",
        "InputFormat": "RecordIO-protobuf", 
        "InputMethod": "tokenized text; training, validation and vocabulary file",
        "MLTraining": "n/a",
        "Regularization": "L1 and L2",
        "Validation": "Accuracy (using validation dataset), BLEU score, Perplexit (cross-entropy)",
        "Preprocessing": "n/a",
        "Hyperparameters": "Batch_size, Learning_Rate, Num_layers_encoder/decoder",
        "TrainingHardware": "Single machine only; GPU only (multiple GPU on single-machine); P3" 
    },
    "DeepAR": {
        "Description": "Forecasting one-dimensional time series data. Uses RNNs; Finds frequencies and seasonality.",
        "MLType": "Time series forecasting (Regression)",
        "InputFormat": "JSON in Gzip or Parquet", 
        "InputMethod": "Start timestamp and Target time series values, dynamic_feat (optional), Cat (optional)",
        "MLTraining": "RNNs",
        "Regularization": "n/a",
        "Validation": "Use entire dataset as test set, remove last time points for training. Evaluate on witheld values.",
        "Preprocessing": "n/a",
        "Hyperparameters": "Context_length, Epochs, mini_batch_size, Learning rate, Num_cells",
        "TrainingHardware": "Single or multi-machine; CPU (recommended), and GPU for training; CPU only for inference. ml.c4.2xlarge/4xlarge" 
    },
    "BlazingText": {
        "Description": "Predict lables for a sentence; machine translation, sentiment analysis.",
        "MLType": "Text classification (Supervised learning); Word2Vec (Unsupervised learning; embeddings)",
        "InputFormat": "TXT, augmented manifest text format", 
        "InputMethod": "One sentence per line; first 'word' is the string '__label__' followed by a label.",
        "MLTraining": "Text classification and Word2Vec",
        "Regularization": "n/a",
        "Validation": "",
        "Preprocessing": "n/a",
        "Hyperparameters": "Text classification: Epochs, Learning_rate, Word_ngrams, Vector_dim; Word2vec: Mode (batch_skipgram, skipgram, cbow), Learning_rate, Window_size, Vector_dim, Negative_samples",
        "TrainingHardware": "Single-machine or GPU for (cbow and skipgram); multiple-CPU for batch_skipgram; Text classification, CPU or GPU; recommend C5 for < 2GB; larger use GPU ml.p2.xlarge or ml.p3.2xlarge" 
    }
}

def main():
    choice = print_menu()
    while choice != 4:
        if choice == 1:
            print_algo("all")
        elif choice == 2:
            algo = input("\nEnter an algorithm: ")
            if algo not in algo_dictionary:
                print("\nInvalid algorithm. Please try again.\n")
                print("Valid algorithm types are: ")
                for key in algo_dictionary:
                    print(key)
                continue
            print_algo(algo)
        elif choice == 3:
            for key in algo_dictionary:
                print(key)
            print()
        choice = print_menu()

    print('\nGood luck!\n')

def print_menu():
    print()
    print("Machine Learning Engineer Certification Training")
    print("================================================\n")
    print("1. Print all algorithms")
    print("2. Print an algorithm")
    print("3. List algorithm names")
    print("4. Exit")
    choice = input("\nEnter your choice: ")
    if choice not in ['1', '2', '3', '4']:
        print("\nInvalid choice. Please try again.\n")
        return print_menu()
    choice = int(choice)
    return choice

def print_algo(algo):
    if algo == "all":
        for key in algo_dictionary:
            print_algo(key)
        return
    else:
        for key, value in algo_dictionary[algo].items():
            print(key, ':', value)
        print()

if __name__ == '__main__':
    main()