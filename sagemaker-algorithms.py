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
    },
    "Object2Vec": {
        "Description": "Create low-dimensional dense embeddings of high-dimensional objects. Like Word2Vec but generalized to handle things other than words",
        "MLType": "Classification; Compute Nearest Neighbors; Cluster Visualization, Genre prediction, Recommendations (similar items or users)",
        "InputFormat": "Tokenize data into integers", 
        "InputMethod": "pairs-of-tokens; i.e. Sentence-sentence, Customer-customer, User-item, Product-product",
        "MLTraining": "CNNs, Bidrectional LSTM, Average-pooled embeddings; Comparator is followed by a feed-forward neural network",
        "Regularization": "n/a",
        "Validation": "",
        "Preprocessing": "n/a",
        "Hyperparameters": "Dropout, early stopping, epochs, learning_rate, batch_size, layers, activation, Enc1_network/Enc2_network--hcnn,bilstm,pooled_embedding",
        "TrainingHardware": "Single-machine; CPU or GPU(multi-GPU OK); ml.m5.2xlarge, ml.p2.xlarge, ml.m5.4xlarge, ml.m5.12xlarge, GPU: P2, P3, G4dn, G5; Inference use: ml.p3.2xlarge; Use INFERENCE_PREFERRED_MODE"
    },
    "Object Dectection": {
        "Description": "Identify all objects in an image with bounding boxes.",
        "MLType": "Classification with MXNet or Tensorflow",
        "InputFormat": "Image as Input. Outputs all instances of objects in the image with categories and confidence scores", 
        "InputMethod": "MXNet: RecordIO or image format (jpg/png); JSON file for annotation",
        "MLTraining": "MXNet: CNN with single-shot multibox Detector (SSD) algorithm, transfer learning mode/incremental training mode; Tensorflow: ResNet, EfficientNet, MobileNet from Tensorflow Model Garden",
        "Regularization": "MXNet: flip, rescale, and jitter for overfitting",
        "Validation": "",
        "Preprocessing": "n/a",
        "Hyperparameters": "Mini_batch_size, Learning_rate, Optimizer: Sgd, adam, rmsprop, adadelta",
        "TrainingHardware": "Single or multiple-machine; GPU instances; ml.p2xlarge/p3, G4dn/G5; Inference: CPU/GPU, M5, P2, P3, G4dn"
    },
    "Image Classification": {
        "Description": "Assign one ore more lables to an image. What objects in an image and not where they are.",
        "MLType": "Classification with MXNet or Tensorflow",
        "InputFormat": "MXNet Default image size is 3-channel (RGB), 224x224 from ImageNet dataset", 
        "InputMethod": "The image",
        "MLTraining": "MXNet: Full Training mode (random weights) or Transfer learning mode (pre-trained weights; top fully-connected layer init with random weights). Tensorflow: MobileNet, Inception, ResNet, EfficientNet, etc. Top layer available for fine-tuning.",
        "Regularization": "n/a",
        "Validation": "n/a",
        "Preprocessing": "n/a",
        "Hyperparameters": "Batch size, learning rate, optimizer; Optimizer-specific: weight decay, beta 1, beta2, eps, gamma",
        "TrainingHardware": "Single and multi-machine; GPU for training; ml.p2/p3, g4dn, g5; multi-GPU OK; Inference: CPU/GPU; M5, P2/P3, G4dn, G5"
    },
    "Semantic Segmentation": {
        "Description": "Pixel-level object classification. Useful for self-driving vehicles, medical imaging diagnostics, robot sensing.",
        "MLType": "Classification",
        "InputFormat": "JPG and PNG annotations (training and validation)", 
        "InputMethod": "Image annotations, label map; augmented manifest image; Pipe mode.",
        "MLTraining": "Built on MXNet Gluon and Gluon CV. Algos: Full-Convolutional Network (FCN), Pyramid Scene Parsing (PSP), DeepLabV3; Backbones: ResNet50, ResNet101, ImageNet",
        "Regularization": "n/a",
        "Validation": "n/a",
        "Preprocessing": "n/a",
        "Hyperparameters": "Epochs, learning rate, batch size, optimizer, etc. Algo and Backbone choice from MLTraining",
        "TrainingHardware": "Single-machine only; GPU only; Training: P2/P3/G4dn, G5; Inference: CPU (C5 or M5) or GPU (P3 or G4dn)"
    },
    "Random Cut Forest": {
        "Description": "Anomaly detection using unsupervised learning. Detected unexpeced spikes in time series data.",
        "MLType": "Classification",
        "InputFormat": "RecordIO-protobuf or CSV", 
        "InputMethod": "File or Pipe mode; time-series data",
        "MLTraining": "decision trees on random sampled data",
        "Regularization": "n/a",
        "Validation": "Optional test channel for computing accuracy, precision, recall, and F1 on labled data (anomaly or not)",
        "Preprocessing": "n/a",
        "Hyperparameters": "Num_trees (increasing reduced noise), Num_samples_per_tree (1/num_samples_per_tree approximates the ratio of anonmalous to normal data if you know ahead of time)",
        "TrainingHardware": "Does not take advantage of GPUs; CPU-only; M4,C4,C5,ML.C5; Inference: ml.c5.xlarge"
    },
    "Neural Topic Model (NTM)": {
        "Description": "(Unsupervised learning) Organize documents into topics; Classify or summarize documents based on topics",
        "MLType": "Topic Modeling",
        "InputFormat": "RecordIO-protobuf or CSV. Four data channels ('train' is required, optional 'validation', 'test', and 'auxiliary')", 
        "InputMethod": "Words must be tokenized into integers. Every document must contain a count for every word in the vocabulary in CSV",
        "MLTraining": "Unsupervised Neurla Variational Inference",
        "Regularization": "n/a",
        "Validation": "n/a",
        "Preprocessing": "n/a",
        "Hyperparameters": "mini_batch_size, learning_rate, num_topics",
        "TrainingHardware": "Does not take advantage of GPUs; CPU-only; M4,C4,C5,ML.C5; Inference: ml.c5.xlarge"
    },
    "Latent Dirichlet Allocation (LDA)": {
        "Description": "(Unsupervised) Topic Modeling not based on deep learning. Can be used for things other than words; E.g. Cluster customers based on purchases; Harmonic analysis in music.",
        "MLType": "Topic modeling",
        "InputFormat": "RecordIO-protobuf or CSV", 
        "InputMethod": "Pipe mode only with recordIO; Each document has counts for every word in vocabulary in CSV format",
        "MLTraining": "Unsupervised. Generates however many topics you specify (non-human-readable topics); Similar to NTM",
        "Regularization": "n/a",
        "Validation": "test channel",
        "Preprocessing": "n/a",
        "Hyperparameters": "Num_topics; Alpha0 - initial guess for concentration parameter; smaller values = sparse topics; larger values (> 1.0) produces uniform mixtures.",
        "TrainingHardware": "Single-machine; CPU Only"
    },
    "K-Nearest-Neighbors (KNN)": {
        "Description": "Classification: Find K closest points to a sample point and return the most frequent label. Regression: Find the K closest points to a sample point and return the average value.",
        "MLType": "Classification and Regression",
        "InputFormat": "Training channel. RecordIO-protobuf or CSV (first column is label)", 
        "InputMethod": "File or pipe mode on either",
        "MLTraining": "Data is sampled, dimensionality reduced, build an index for looking up neighbors, serialize the mmodel, and query the model for a given K.",
        "Regularization": "sign or fjlt methods to avoide 'curse of dimensionality' at a cost of noise/accuracy",
        "Validation": "Test channel: emits accuracy or MSE",
        "Preprocessing": "n/a",
        "Hyperparameters": "K!, Sample_zie",
        "TrainingHardware": "Training: CPU/GPU: ml.m5.2xlarge; ml.p2.xlarge; CPU for lower latency; GPU for higher throughput on large batches"
    },
    "K-Means Clustering": {
        "Description": "Unsupervised clustering. Every observation mapped to n-dimensional space (n = number of features).",
        "MLType": "Clustering, classification",
        "InputFormat": "RecordIO-protobuf or CSV", 
        "InputMethod": "File or Pipe on either",
        "MLTraining": "Clustering measured by Euclidean distance. K-means++; Training uses ShardedByS3Key; Testing uses FullyReplicated.",
        "Regularization": "n/a",
        "Validation": "test channel",
        "Preprocessing": "n/a",
        "Hyperparameters": "K!, Mini_batch_size, Extra_center_factor, Init_method",
        "TrainingHardware": "CPU/GPU; CPU recommended; One GPU per instance; GPU: ml.g4dn.xlarge, P2, P3, G4dn, G4"
    },
    "Principal Component Analysis (PCA)": {
        "Description": "Unsupervised Dimensionality reduction technique for taking higher-dimensional data and refining into lower-dimensional data (like a 2D plot), while minimizing loss of information.",
        "MLType": "Dimensionality reduction",
        "InputFormat": "RecordIO-protobuf", 
        "InputMethod": "File or Pipe on either",
        "MLTraining": "Covariance matrix then singular value decomposition (SVD); Regular mode: for sparse data and moderate observations/features; Randomized mode: for large number of obersations/features and uses approximation algorithms.",
        "Regularization": "n/a",
        "Validation": "n/a",
        "Preprocessing": "n/a",
        "Hyperparameters": "Alogrithm_mode, Subtract_mean (unbias data)",
        "TrainingHardware": "GPU/CPU"
    },
    "Factorization Machines": {
        "Description": "Pair-wise interactions recommender system: Click prediction, Item recommendations, with sparse data (users don't interact with most pages).",
        "MLType": "(Supervised) Classification or Regression",
        "InputFormat": "RecordIO-protouf (float32)", 
        "InputMethod": "n/a",
        "MLTraining": "Recommender Systems: Factorization methods on Matrices for recommendations",
        "Regularization": "n/a",
        "Validation": "n/a",
        "Preprocessing": "n/a",
        "Hyperparameters": "Initialization methods for bias, factors, and linear terms: uniform/normal/constant; Can tune properties of each method.",
        "TrainingHardware": "CPU/GPU: CPU recommended, GPU only works on dense data"
    },
    "IP Insights": {
        "Description": "Identifies suspicious behavior from IP address usage patterns.",
        "MLType": "",
        "InputFormat": "CSV only (Entity,IP)", 
        "InputMethod": "Usernames, Account IDs; Training channel",
        "MLTraining": "Neural Network to learn latent vector representations of entities and IP addresses; large hashes; generates negative samples during training by randomly pairing entities-IPs.",
        "Regularization": "n/a",
        "Validation": "Optional validation which computes AUC score",
        "Preprocessing": "None",
        "Hyperparameters": "Epochs, learning rate, batch size, Num_entity_vectors (hash size, set twice number of entity identifiers), Vector_dim (size of embedding vectors, too large = overfitting)",
        "TrainingHardware": "CPU/GPU (GPU recommended); multiple GPUs; ml.p3.2xlarge or higher; CPU size depends on vector_dim and num_entity_vectors"
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