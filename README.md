# 📰 Fighting Misinformation: Source-Based Fake News Classification 🕵️‍♂️

Fake news spreads like wildfire in today’s fast-paced digital world. This project focuses on **classifying news by type and label** using a source-based approach. By leveraging structured data and machine learning, we aim to combat misinformation and bring transparency to online news.

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.


## 📊 About the Dataset
### 🔎 Context
Social media platforms are a treasure trove of content, with **news** being one of the most consumed categories. However, not all news is authentic. Fake news, whether posted by politicians, news outlets, or civilians, can have far-reaching consequences. 

**Challenges**:
- Manual classification of news is **time-consuming** and prone to **bias**.
- Verifying authenticity remains a critical task in the fight against misinformation.

### 🔒 Source
Published paper: [Source-Based Fake News Classification](http://www.ijirset.com/upload/2020/june/115_4_Source.PDF)

### 🔧 Features
- Preprocessed data from the **Getting Real about Fake News** dataset.
- Eliminated skew for improved reliability.
- Comprehensive inclusion of source information, including author names, publication dates, and labels.

## 🚀 Motivation
In an age where fake WhatsApp forwards and misleading Tweets influence public opinion, it’s crucial to develop tools to:
- Mitigate the spread of misinformation.
- Inform users about the nature of the news they consume.

This project’s inspiration lies in creating:
1. **Practical applications** to analyze and classify news articles.
2. **Plugins** and tools for easy access to fact-checking.
3. **Awareness** campaigns about the consequences of consuming and spreading fake news.

## 🌟 Highlights
- **Source-Based Labeling**: Ensures credibility by tracking the origin of news articles.
- **Automation**: Reduces human bias in classification.
- **Informed Consumption**: Helps users make smarter decisions about the news they trust.

## ⚖️ Comparison of Approaches

In this project, two machine learning approaches are evaluated for classifying fake news:

1. **RandomForestClassifier using TF-IDF**
2. **Embeddings + CNN (Convolutional Neural Networks)**

### 1. **RandomForestClassifier using TF-IDF**
- **TF-IDF** (Term Frequency-Inverse Document Frequency) is a traditional text preprocessing technique that transforms text data into a high-dimensional sparse vector space. This method measures the importance of a word in a document relative to its frequency across all documents.
- The **RandomForestClassifier** then uses this vectorized representation for classification. Random forests are an ensemble method that builds multiple decision trees and combines their outputs, typically resulting in a strong and reliable classifier.
  
   **Pros**:
   - Efficient and works well for smaller datasets.
   - Simple to implement and interpret.
  
   **Cons**:
   - The sparse representation of text doesn’t capture the semantic meaning of words or their contextual relationships.
   - May struggle with large datasets or when the relationships between words are complex.

### 2. **Embeddings + CNN (Convolutional Neural Networks)**
- **Embeddings** are dense, lower-dimensional vector representations of words that capture their semantic meaning. By mapping words with similar meanings closer together in a vector space, embeddings provide more context and depth compared to traditional vectorization methods like TF-IDF.
- The **CNN** architecture is well-suited for text classification tasks. In this case, convolutional layers capture local patterns in the text, and pooling layers help reduce dimensionality. CNNs can learn more abstract and hierarchical features from text, which is useful in identifying subtle patterns and relationships that might indicate whether news is fake or real.
  
   **Pros**:
   - Better at capturing semantic relationships and context of words.
   - Suitable for large and complex datasets with nuanced patterns.
   - Can provide higher performance in text classification tasks.
  
   **Cons**:
   - Requires larger datasets for training.
   - Needs more computational resources and may take longer to train.

### **Model Evaluation**
- Both approaches were trained and evaluated on the **Getting Real about Fake News** dataset.
- The **RandomForestClassifier using TF-IDF** showed decent performance for basic tasks but struggled to capture deeper semantic meaning and context.
- The **Embeddings + CNN** approach outperformed the traditional method in both training and testing accuracy, as it was able to better capture the relationships between words and classify news more effectively.

### Conclusion
The results of this comparison highlight the advantages of using **Embeddings + CNN** for more complex text classification tasks, especially in dealing with large, high-dimensional datasets. However, **RandomForestClassifier using TF-IDF** remains a useful and simpler tool for tasks where computational resources or training data are limited. This project shows that using a source-based approach combined with machine learning techniques can effectively aid in the detection of fake news.

## 🙏 Acknowledgements
- Dataset: **Getting Real about Fake News**
  - Selected for its detailed inclusion of source information, crucial for verifying authenticity.
- Special thanks to the creators and contributors of this dataset for enabling research in combating misinformation.

✨ Let’s Fight Fake News Together! 🕵️‍♂️


https://www.kaggle.com/datasets/ruchi798/source-based-news-classification/data
