#importing modules
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model
import time
import matplotlib.pyplot as plt
import numpy as np
import keras as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import networkx as nx
from scipy.linalg import fractional_matrix_power

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import requests
import pytz
from bs4 import BeautifulSoup
import datetime
import csv

#verifying the certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#pre-processing target domain

class YTRecommender():
    def __init__(self):
        pass

    def preprocessTarget(self,filepath):
        
        global df_target
        # Read the CSV file into a Pandas DataFrame
        df_target = pd.read_csv(filepath)
        #here we will only be using 'title' and 'video.tags' for our purpose
        #if you want to do DANN, use 'description' column as well so that the recipe & description are relevant to have similar features
        #so we will be dropping the remaining columns
        df_target = df_target.drop(columns=['q', 'queryTime', 'description','rank', 'publishedAt', 'channelTitle', 'totalResults', 'kind', 'channelId', 
                                            'default.height', 'default.url', 'default.width', 'high.height', 'high.url', 'high.width', 
                                            'liveBroadcastContent', 'medium.height', 'medium.url', 'medium.width', 'nextPageToken', 'playlistId', 'resultsPerPage', 'thumbnails', 'videoId', 'video.contentDetails', 
                                            'video.etag', 'video.id', 'video.kind', 'video.localizations', 'video.player', 'video.recordingDetails', 'video.snippet', 'video.statistics', 'video.status', 'video.topicDetails', 'video.categoryId', 'video.channelId', 'video.channelTitle', 'video.defaultAudioLanguage', 'video.defaultLanguage', 'video.description', 'video.liveBroadcastContent', 'video.localized', 'video.publishedAt', 'video.thumbnails', 'video.title', 'video.relevantTopicIds', 'video.topicCategories', 'video.topicIds', 'video.commentCount', 'video.dislikeCount',
                                            'video.favoriteCount', 'video.likeCount', 'video.viewCount', 'video.embeddable', 'video.license', 'video.privacyStatus', 'video.publicStatsViewable', 'video.uploadStatus', 'video.caption', 'video.definition', 'video.dimension', 'video.duration', 'video.licensedContent', 'video.projection', 'video.regionRestriction', 'video.liveStreamingDetails', 'video.contentRating', 'channel.contentDetails', 'channel.etag', 'channel.id', 'channel.kind', 'channel.snippet', 'channel.statistics', 'channel.country', 
                                            'channel.customUrl', 'channel.defaultLanguage', 'channel.description', 'channel.localized', 'channel.publishedAt', 'channel.thumbnails', 'channel.title', 'channel.commentCount', 'channel.hiddenSubscriberCount', 'channel.subscriberCount', 'channel.videoCount', 'channel.viewCount', 'channel.relatedPlaylists'])
        # Assuming df['video.tags'] is your text data
        df_target['video.tags'].fillna('', inplace=True)  # Replace NaN values with an empty string
        
        #here we are creating a labels for target domain because the dataset does not have labels
        #1/3 part of the labels to 'appetizers'. another 1/3 part to 'dinner' and the last 1/3 part to 'desserts'

        total_size = len(df_target)
        category_size = total_size // 3
        df_target['labels'] = None
        df_target.loc[:category_size - 1, 'labels'] = 'Appetizers'
        df_target.loc[category_size:2*category_size - 1, 'labels'] = 'Dinner'
        df_target.loc[2*category_size:total_size - 1, 'labels'] = 'Desserts'
        
        #shuffling the dataframe
        df_target = df_target.sample(frac=1, random_state=42)
        return df_target
    
    def create_topic_space(self,recipe_text):
    # Tokenize the text
        words = word_tokenize(recipe_text.lower())  # Convert to lowercase for consistency

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

        # Define the number of main words to extract
        num_main_words = 7

        # Get the most common words as main words
        word_freq = nltk.FreqDist(filtered_words)
        main_words = [word for word, _ in word_freq.most_common(num_main_words)]

        return main_words
        
    
    def preprocessAuxiliary(self,filepath):
        
        global df_source
        # Read the CSV file into a Pandas DataFrame
        df_source = pd.read_csv(filepath)
        #here we will only be using 'Ingredients' for our purpose
        #so we will be dropping the remaining columns
        df_source = df_source.drop(columns=['Unnamed: 0', 'Ingredients', 'Image_Name',
            'Cleaned_Ingredients','Title'])
        # Assuming df['Instructions'] is your text data
        df_source['Instructions'].fillna('', inplace=True)  # Replace NaN values with an empty string
        # Create topic space
        df_source['topic_space'] = None
        for i,recipe in enumerate(df_source['Instructions']):
            topic_space = self.create_topic_space(recipe)
            df_source['topic_space'][i] = topic_space
            
        # Assuming df['topic_space'] is your text data
        df_source['topic_space'].fillna('', inplace=True)  # Replace NaN values with an empty string
        return df_source
    

    # Define a function to generate dummy labels
    def generate_dummy_labels(self, num_samples):
        dummy_labels = ["X"] * num_samples
        df_source['labels'] = dummy_labels
        return df_source

    num_samples = len(df_source)

    # Generate dummy labels for EMNIST
    generate_dummy_labels(num_samples)
    
    def prepLabels(self):
        global domain_label_source,domain_label_target
        # Prepare domain labels
        domain_label_source = np.zeros((df_source['topic_space'].shape[0], 1))  # Source domain label is 0
        domain_label_target = np.ones((df_target["video.tags"].shape[0], 1))   # Target domain label is 1
        
    def concat(self):
        
        global x_combined,y_combined,domain_labels_combined
        
        # Concatenate source and target data
        x_combined = df_target['video.tags'] + df_source['topic_space']
        y_combined = df_target['labels'] + df_source['labels']
        domain_labels_combined = np.vstack((domain_label_source, domain_label_target))
        
        #since the size of domain labels is more than x_combined/y_combined, we will have the domain_labels_combined of the same size
        domain_labels_combined = domain_labels_combined[:13501]
        
        # Assuming x_combined is your text data
        x_combined.fillna('', inplace=True)  # Replace NaN values with an empty string

        # Assuming y_combined is your text data
        y_combined.fillna('', inplace=True)  # Replace NaN values with an empty string

    '''
            Graph Based framework and spectral learning through eigen vectors in laplacian matrix:
    
    

            To implement a graph-based framework and spectral learning in your code, 
            we'll make use of the NetworkX library for graph operations and NumPy for 
            linear algebra operations. The approach involves constructing a graph, computing the 
            Laplacian matrix, performing eigen decomposition, and utilizing the eigenvectors for 
            feature representation.
    '''
        
    def graph(self):
        # Constructing a Graph-Based Framework
        corpus = x_combined.apply(lambda x: set(x.split()))  # Convert each tag set to a set

        global G
        # Build graph from corpus using Jaccard similarity
        G = nx.Graph()
        for i, tags_i in enumerate(corpus):
            G.add_node(i, tags=tags_i)  # Node represents a video with its tags
            for j, tags_j in enumerate(corpus):
                if i != j:
                    intersection_size = len(tags_i.intersection(tags_j))
                    union_size = len(tags_i.union(tags_j))
                    if union_size > 0:
                        jaccard_similarity = intersection_size / union_size
                        if jaccard_similarity > 0.2:  # Adjust the threshold as needed
                            G.add_edge(i, j, weight=jaccard_similarity)
                            
    def plot(self):
        pos = nx.spring_layout(G)  # Positioning nodes using spring layout algorithm

        # Plotting nodes
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=50)

        # Plotting edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

        # Displaying labels (optional)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        # Display the plot
        plt.title("Graph Visualization")
        
        return plt.show()

    def linearAlg(self):
        
        # Laplacian matrix
        L = nx.laplacian_matrix(G).toarray()

        # Spectral Learning and Eigen Feature Representations

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select dominant eigenvectors (adjust the number as needed)
        global num_eigenvectors
        num_eigenvectors = 10
        dominant_eigenvectors = eigenvectors[:, :num_eigenvectors]

        # Eigen Feature Representations
        global eigen_feature_representations
        eigen_feature_representations =fractional_matrix_power(L, -0.5) @ dominant_eigenvectors


    #Here, we will be tokenizing the sentences.
    
    def tokenize(self):
        #defining the parameters
        
        global vocab_size,embedding_dim,max_sequence_length, num_classes
        num_classes = 1
        embedding_dim = 100

        # Assuming df['video.tags'] is your text data, we  tokenize the input dataset into tokens for the CNN model
        #Tokenizer Initialization and Fitting:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x_combined)

        #Vocabulary Size and Maximum Sequence Length Calculation:
        
        vocab_size = len(tokenizer.word_index) + 1
        max_sequence_length = max(x_combined.apply(lambda x: len(x.split())))

        #Texts to Sequences:
        sequences = tokenizer.texts_to_sequences(x_combined)

        #Padding Sequences
        data = pad_sequences(sequences, maxlen=max_sequence_length)
        
        
        # Assuming 'text' is your input data and 'label' is your target variable
        X = data   #data
        y = y_combined.values           #labels

        # Convert labels to numerical format using LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Integrate with Cross-Domain Transfer Learning
        # Concatenate eigen feature representations with existing data
        global X_train_with_graph, X_test_with_graph
        X_train_with_graph = np.hstack((X_train, eigen_feature_representations[:len(X_train)]))
        X_test_with_graph = np.hstack((X_test, eigen_feature_representations[len(X_train):]))

        
    #building a DANN
    
    # Define the Feature Extractor
    def build_feature_extractor(self,vocab_size,embedding_dim,max_sequence_length):
        model = models.Sequential([
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
            layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
            layers.GlobalMaxPooling1D()
        ])
        return model
    
    # Define the Label Predictor
    def build_label_predictor(self,num_classes):
        model = models.Sequential([
            Dense(units=num_classes, activation='softmax',name='output')
        ])
        return model
    
    # Define the Domain Predictor
    def build_domain_predictor():
        model = models.Sequential([
            layers.Dense(64,activation='relu',name='dense_1'),
            layers.Dense(1,activation='sigmoid',name='dense_2')
        ])
        return model
    
    # Build the complete DANN model
    def build_dann(self,vocab_size,embedding_dim,max_sequence_length, num_classes,num_eigenvectors):
        feature_extractor = self.build_feature_extractor(vocab_size,embedding_dim,max_sequence_length)
        label_predictor = self.build_label_predictor(num_classes)
        domain_predictor = self.build_domain_predictor()

        # Define inputs
        input_data = layers.Input(shape=(max_sequence_length + num_eigenvectors,))
        label = layers.Input(shape=(num_classes,))
        domain_label = layers.Input(shape=(1,))

        # Feature extractor output
        feature_output = feature_extractor(input_data)

        # Label prediction branch
        label_output = label_predictor(feature_output)

        # Domain prediction branch
        domain_output = domain_predictor(feature_output)

        dann_model = models.Model(inputs=[input_data, label, domain_label],
                                outputs=[label_output, domain_output])

        return dann_model
    
        # Define loss functions
    def label_loss(self,y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    def domain_loss(self,y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)

    def compileTrain(self):
        label_loss = self.label_loss()
        domain_loss = self.domain_loss()
        # Build and compile the DANN model        
        dann_model = self.build_dann(vocab_size,embedding_dim,max_sequence_length, num_classes,num_eigenvectors)

        dann_model.compile(optimizer='Adam',
                        loss=[label_loss, domain_loss],
                        loss_weights=[1.0,0.1])
        
            # Train the DANN
        global start_time,end_time
        start_time = time.time()

        dann_model.fit([X_train_with_graph, y_train, domain_labels_combined[:len(X_train)]],
                                [y_train, domain_labels_combined[:len(X_train)]],
                                epochs=1)

        end_time = time.time()

    def realtimeScrape(self):
        r = requests.get('https://www.allrecipes.com/gallery/top-new-recipes-2022/')
        html = r.content 
        soup = BeautifulSoup(html,'html.parser')
        
        for w in soup.find_all('p', {'class': 'comp mntl-sc-block mntl-sc-block-html'})[:]:
            model_input = self.create_topic_space(w.text)
                        
        return model_input
        
    
    
    
    
    

        