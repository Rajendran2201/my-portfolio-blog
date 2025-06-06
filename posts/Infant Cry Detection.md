# Infant Cry Detection - a deep learning based approach to analyse baby cry sounds

Every mother would have experienced the moment where they couldn't console their baby from crying. This would have been a reason for a numerous sleepless nights fro the mothers. Whether it’s hunger, discomfort, or just the need for comfort, understanding the reason behind a baby’s cry is crucial for both the child’s well-being and the parent’s peace of mind. Many parents undergo a phase of mental distortness due the feeling of not taking good care of their baby. 

In the recent times, the society is more driven towards nuclear families and parents can't afford to spend their entire day with the baby, especially when both of them are working parents. They tend to look for a better or modern parenting approach. 

TinyToes is one such software product that avails the parents to take care of their child in a modern way. TinyToes is an AI-powered parenting platform designed to provide first-time mothers and new parents with real-time guidance, expert insights, and personalized support. It features an AI-driven Baby Cry Analyzer to interpret a baby’s needs, 24/7 Expert Consultation, a Milestone Tracker for growth and vaccinations, and Emergency Assistance for instant medical support. A Mom’s Corner fosters a supportive community, while a Smart AI Chatbot offers instant parenting advice. With seamless access to trusted pediatricians and childcare services, TinyToes simplifies parenting by integrating all essential baby care tools into one platform, reducing anxiety and ensuring a stress-free parenting journey.

 ![](https://i.imgur.com/hj5N09V.png)

![](https://i.imgur.com/g0QCUcw.png)


We had several features and modules in TinyToes among which the most prominent one was the cry detection part. Imagine how cool it would if you were able to actually decode what a child is trying to say, which is exactly what TinyToes does. 

> To further know about the project, click [here](https://github.com/Rajendran2201/babycare/tree/main)

In this article, We'll discuss how we built the model for decoding and analysing the baby cry sounds. 

We'll begin from the data. Of course, we need data, a lot of data when we try to build something like this. In our case, we had access to [donateacry-corpus](https://github.com/gveres/donateacry-corpus) data set which is a open source dataset collected by a group of volunteers. The dataset contained of 457 audio files. The audio files contain baby cry samples, with the corresponding labels identifying the cry reason. There are 457 audio files of babies crying classified into belly_pain (16 files), burping (8 files), discomfort (27 files), hungry (382 files) and tired (24 files). This was the dataset that we used for training our deep learning model.

---

As we collected the dataset, we moved on to the next process. Here, the entire process of building a model is classified into two stages: 
1. Data preprocessing
2. Model Training

## Data Preprocessing

As the dataset was full of audio files, we cannot train a model with these data. Generally, the different audio data have different kind of sampling rates. We cannot train a model with the audio data with different sampling rates. Even if we do, we cannot assure the reliability of the model. So, the general practice is to convert the audio files into some sort of images and then, train the model with these generated images accordingly. This would predominantly eliminate the differences in the sampling rates of the audio files. 

#### So, How can we do this?

Firstly, we collected the audio files and try to convert them into images. The audio is non-stationary in nature. They need to e processed to be converted into images. The audio is initially divided into smaller parts called short-time frames. This process is called as frame segmentation. This is done with the help of **Hann Window**. The Hann Window $w(n)$ is primarily used for signal filtering and signal analysis.

$$w(n) = \frac{1}{2} - \frac{1}{2}\cos(\frac{2\pi n}{N-1})$$

The audio files after frame segmentation has to be transformed into spectrograms using Short Time Fourier Transformation (STFT). 

**What is a Spectrogram?**

A spectrogram is **a visual representation of the spectrum of frequencies of a signal as it varies with time**. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams. When the data are represented in a 3D plot they may be called waterfall displays.

A spectrogram would probably look like this: 

![](https://i.imgur.com/KH6slAA.png)


![](https://i.imgur.com/5jUQ54s.png)


**What does a spectrogram mean?**

- Spectrogram is a way of representing an audio as an image. 
- The $X-axis$ represents the time and $Y-axis$ represents the frequency.
- The colors of the image denotes the magnitude of the spectral energy density.
- The top portion of the image represents the higher frequency sounds and the bottom portion represents the lower frequency sounds.

![](https://i.imgur.com/Db2JCCs.png)

Once the spectrograms have been derived from the audio files, we can use them for the model training. In order to achieve a better performance, we can preprocess them a bit more. 

#### Mel Spectrograms

- The spectrograms are then transformed into mel spectrograms through mel scaling.
- The spectrograms are direct reflection of the audio files. This may contain some frequencies which are not recognised by the human auditory system. 
- We have to convert the images into a form which is compatible with the human auditory system, which is why we convert the spectrograms into mel spectrograms.

#### Mel Frequency Cepstral Co-efficients (MFCC)

- The mel spectrograms are then converted into Mel Frequency Cepstral Co-efficients (MFCC).
- This is done through the process called Discrete Cosine Transformation (DCT).
- This transformation majorly focuses on extracting the main envelope information from the mel spectrograms. 
- This helps in ignoring the irrelevant information and focus the important features.

![](https://i.imgur.com/pBMdixV.png)

#### What are we gonna do with this?

Now, we have extracted (1) Spectrograms (2) Mel Spectrograms and (3) Mel Frequency Cepstral Coefficients from the data. We have almost completed the preprocessing stage of the data. 

It's time to train the model with our preprocessed data. But, here we are not going to train the model with just the MFCC, instead we are going for a different approach. 

We are going to create a hybrid image comining all these processed images: (1) Spectrograms (2) Mel Spectrograms and (3) Mel Frequency Cepstral Coefficients and the model is then trained on the multi dimensional hybrid features extracted from the hybrid images. By combining MFCC with the processed spectrogram and mel spectrogram features, the resulting hybrid feature set provides a richer representation of the audio data, capturing both spectral and cepstral characteristics.

This kind of model training with hybrid features,
- Improves the feature set for tasks like **speech recognition, speaker identification, and audio classification** by leveraging the strengths of different representations.
- Reduces redundancy while preserving essential information.
- Enhances performance in machine learning models by providing a more comprehensive feature space.


![](https://i.imgur.com/IgkowfD.png)

## Model Training

Going with the traditional approach of learning will not yield a fruitful result. Thus, we have used a modified two level learning strategy. The hybrid representing the audios have two kinds of features: local features and global features.

Using deep learning architectures such as ResNet50 with Multihead Attention, the model learns to differentiate between various cry categories, such as hunger, pain, discomfort, or sleepiness. Training requires a large and diverse dataset to ensure robust performance across different babies.

### What are these features?

1. **Local Features**
    - Represent **small-scale patterns**, such as:
        - Short-term spectral changes.
        - Local frequency variations.
        - Fine-grained textures in spectrograms.
    - **Local features** help identify **small, intricate patterns** crucial for distinguishing sounds at a low level.
    - Essential for identifying **phonemes, syllables, or short-term sound characteristics** in speech/audio data.

2. **Global Features**
    - Represent **long-range dependencies and overall structure**, such as:
        - The temporal relationships across an entire audio sequence.
        - The context of a spoken phrase rather than just individual phonemes.
    - **Global features** ensure the model understands the **broader context**, such as how sounds evolve over time.
    - Helps in understanding **word sequences, speaker characteristics, and emotion detection** in speech/audio.

Learning both **local** and **global** features enhances the model’s ability to capture **fine-grained details** and **overall context** in the data. This dual approach is especially useful in tasks like **speech recognition, audio classification, and signal processing**, where both **short-term variations** and **long-term dependencies** matter. Therefore, The model has to undergo both local and global feature learning to completely analyse the patterns in the data.


![](https://i.imgur.com/gpntXsu.png)

---
## Local Feature Learning Layer

In the Local Feature Learning Layer, The model will be trained on the local features such as short-term acoustic elements. This is done by decomposing the hybrid images into smaller modules named as **Local Feature Extraction Block** (LFEB).

The **Local Feature Extraction Block (LFEB)** consists of the following layers:

1. **Convolutional Layer**
    - Applies filters to the input feature maps to extract **local patterns** like edges, textures, and frequency variations in audio data.
    - Helps the model learn hierarchical representations from raw spectrograms or MFCCs.

2. **ReLU (Rectified Linear Unit) Activation Function**
    - Introduces **non-linearity** into the model, allowing it to learn complex patterns.
    - Prevents the vanishing gradient problem by converting negative values to zero and keeping positive values unchanged.

3. **Dropout Layer**
    - Randomly drops (deactivates) a percentage of neurons during training to **prevent overfitting**.
    - Improves model generalization by forcing it to learn more robust features instead of memorizing training data.

4. **Batch Normalization (BN) Layer**
    - Normalizes the inputs to maintain a stable distribution of activations.
    - Speeds up training by reducing **internal covariate shift**, making optimization more efficient.
    - Helps combat overfitting by introducing slight regularization.

5. **Pooling Layer**
    - Reduces the spatial dimensions of the feature maps, making the model more **computationally efficient**.
    - Helps retain essential features while discarding unnecessary details.
    - Common pooling methods:
        - **Max pooling:** Retains the most significant features.
        - **Average pooling:** Takes the mean values, leading to smoother feature representations.

By stacking four such blocks, the **network enhances local feature extraction**, improving the model's ability to recognize patterns in the audio signal. This structure **reduces computational cost, prevents overfitting, and accelerates training** while capturing essential information.

![](https://i.imgur.com/gcG9bql.png)

---
## Global Feature Learning Layer

The global feature learning network consists of,
1. Multihead attention layer
2. Long Short Term Memory (LSTM) layer
3. Softmax layer

#### Multihead Attention Layer

The **Multi-Head Attention (MHA)** mechanism is a key component in **Transformer models** (like in BERT and GPT) and is widely used for processing sequential data, including speech and audio. It allows the model to **focus on different parts of the input simultaneously**, capturing **global dependencies** efficiently.

- Traditional **CNNs capture local patterns**, but they struggle with long-range dependencies.
- **RNNs/LSTMs** can process sequential data, but they are slow and have difficulty handling very long sequences.
- **Multi-Head Attention** solves both problems by enabling parallel computation and **learning relationships across the entire sequence** in a single step.

The Multi-Head Attention mechanism takes an input sequence and transforms it using **three key matrices**:

- **Query (Q)** → Represents the current position looking for relevant information.
- **Key (K)** → Represents all positions in the input sequence with which the query compares.
- **Value (V)** → Holds the actual data content that will be passed forward.

These matrices are used to compute **attention scores**, determining which parts of the input are most relevant to each position.

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
#### LSTM Layers

LSTMs are a type of **Recurrent Neural Network (RNN)** designed to handle **sequential data** like **audio, speech, time series, and natural language**. Unlike standard RNNs, LSTMs can **learn long-term dependencies** and avoid the **vanishing gradient problem** using a special gating mechanism.

Traditional RNNs suffer from **short-term memory issues**—they struggle to retain information over long sequences because gradients shrink (vanishing gradient problem). **LSTM solves this by introducing memory cells and gates**, allowing it to **retain and forget** information selectively over long time steps.

At each time step $t$:
1. **Forget Gate** decides which past information to keep/drop.
2. **Input Gate** updates memory with new input.
3. **Cell State** is updated with both old and new information.
4. **Output Gate** decides what to send to the next time step.

Final **hidden state $h_t$ and **cell state $C_t$ are passed to the next step.

$$h_t=o_t⋅tanh⁡(C_t)$$
#### Softmax Layer

- The **Softmax layer** is commonly used in the final layer of classification models. It **converts raw logits (unnormalized outputs)** into **probability distributions** across multiple classes. 
- Softmax is a normalization function used to normalize the weights such that the sum of the scores would result as 1.

By this approach, the model can be trained on the data to predict the infant cry sounds. We have achieved 88% of accuracy on using this approach. Further, on the usage of 10-fold cross validation, the accuracy even increased on the test data.

Infant cry detection powered by deep learning is transforming how parents and caregivers understand and respond to a baby’s needs.  While every baby is unique, connecting with other mothers and sharing experiences can be incredibly helpful and TinyToes definitely bridges that gap and drives the parents towards modern parenting.

