# 📰 Fighting Misinformation — A Personal Learning Project on Fake News Classification

This project was developed as part of my hands-on journey through a **Deep Learning course**, where I focused on understanding how AI can be applied to real-world challenges—particularly the spread of **fake news** across digital platforms.

In today's fast-paced digital world 🌐, misinformation spreads rapidly, often distorting public perception and influencing critical decisions. From shaping elections 🗳️ to triggering widespread panic during crises, the consequences are real and significant.

To explore this issue from a practical and technical perspective, I built a machine learning model that classifies news articles as **real or fake**, using a **source-based approach**. Rather than analyzing article content directly, the model looks at structured metadata—such as:

- The **author** of the article ✍️  
- The **publication date** 📅  
- And the **reputation of the source** itself 🏅

By training on this structured data, the goal was to simulate how credibility might be algorithmically assessed based on source characteristics alone. This approach reflects a more **explainable and efficient** path to automated verification systems, while also encouraging deeper awareness of where our information comes from.

> 🤖 Tools & Techniques: I implemented this using **scikit-learn** and **Pandas**, exploring different classifiers (e.g., Logistic Regression, Random Forests), and working through typical steps like preprocessing, feature selection, and evaluation with metrics like accuracy and F1-score.

More than just a technical exercise, this project reinforced my understanding of how machine learning models can assist in **digital literacy**, promoting more **informed and critical media consumption**.

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

🔍 This project was inspired by the Kaggle notebook [*EDA and Modelling on News Dataset (99% accuracy)*](https://www.kaggle.com/code/bansalvishesh/eda-and-modelling-on-news-dataset-99-accuracy) by **Vishesh Bansal**, which provided valuable insights into text preprocessing and classification workflows.

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

## ⚠️ Disclaimer  
**This project was developed for educational and research purposes only.** It is an academic exploration of **machine learning techniques for source-based fake news classification**.  

The models and techniques presented in this repository **are not intended for real-world misinformation detection or journalistic verification**. They serve as a proof of concept and have not been extensively tested for accuracy, bias, or robustness in diverse media environments.  

While this project leverages publicly available datasets and references existing research, **users should not rely on its outputs for making factual or editorial decisions**. Always verify news from multiple trusted sources.  

## 🌟 Explore My Other Cutting-Edge AI Projects! 🌟

If you found this project intriguing, I invite you to check out my other AI and machine learning initiatives, where I tackle real-world challenges across various domains:

+ [🌍 Advanced Classification of Disaster-Related Tweets Using Deep Learning 🚨](https://github.com/sergio11/disasters_prediction)  
Uncover how social media responds to crises in real time using **deep learning** to classify tweets related to disasters.

+ [📰 Fighting Misinformation: Source-Based Fake News Classification 🕵️‍♂️](https://github.com/sergio11/fake_news_classifier)  
Combat misinformation by classifying news articles as real or fake based on their source using **machine learning** techniques.

+ [🛡️ IoT Network Malware Classifier with Deep Learning Neural Network Architecture 🚀](https://github.com/sergio11/iot_network_malware_classifier)  
Detect malware in IoT network traffic using **Deep Learning Neural Networks**, offering proactive cybersecurity solutions.

+ [📧 Spam Email Classification using LSTM 🤖](https://github.com/sergio11/spam_email_classifier_lstm)  
Classify emails as spam or legitimate using a **Bi-directional LSTM** model, implementing NLP techniques like tokenization and stopword removal.

+ [💳 Fraud Detection Model with Deep Neural Networks (DNN)](https://github.com/sergio11/online_payment_fraud) 
Detect fraudulent transactions in financial data with **Deep Neural Networks**, addressing imbalanced datasets and offering scalable solutions.

+ [🧠🚀 AI-Powered Brain Tumor Classification](https://github.com/sergio11/brain_tumor_classification_cnn)  
Classify brain tumors from MRI scans using **Deep Learning**, CNNs, and Transfer Learning for fast and accurate diagnostics.

+ [📊💉 Predicting Diabetes Diagnosis Using Machine Learning](https://github.com/sergio11/diabetes_prediction_ml)  
Create a machine learning model to predict the likelihood of diabetes using medical data, helping with early diagnosis.

+ [🚀🔍 LLM Fine-Tuning and Evaluation](https://github.com/sergio11/llm_finetuning_and_evaluation)  
Fine-tune large language models like **FLAN-T5**, **TinyLLAMA**, and **Aguila7B** for various NLP tasks, including summarization and question answering.

+ [📰 Headline Generation Models: LSTM vs. Transformers](https://github.com/sergio11/headline_generation_lstm_transformers)  
Compare **LSTM** and **Transformer** models for generating contextually relevant headlines, leveraging their strengths in sequence modeling.

+ [🩺💻 Breast Cancer Diagnosis with MLP](https://github.com/sergio11/breast_cancer_diagnosis_mlp)  
Automate breast cancer diagnosis using a **Multi-Layer Perceptron (MLP)** model to classify tumors as benign or malignant based on biopsy data.

+ [Deep Learning for Safer Roads 🚗 Exploring CNN-Based and YOLOv11 Driver Drowsiness Detection 💤](https://github.com/sergio11/safedrive_drowsiness_detection)
Comparing driver drowsiness detection with CNN + MobileNetV2 vs YOLOv11 for real-time accuracy and efficiency 🧠🚗. Exploring both deep learning models to prevent fatigue-related accidents 😴💡.

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

## ⚠️ Disclaimer  
**This project was developed for educational and research purposes only.** It is an academic exploration of **machine learning techniques for source-based fake news classification**.  

The models and techniques presented in this repository **are not intended for real-world misinformation detection or journalistic verification**. They serve as a proof of concept and have not been extensively tested for accuracy, bias, or robustness in diverse media environments.  

While this project leverages publicly available datasets and references existing research, **users should not rely on its outputs for making factual or editorial decisions**. Always verify news from multiple trusted sources.  

✨ Let’s Fight Fake News Together! 🕵️‍♂️

## **🙏 Acknowledgments**

- Dataset: **Getting Real about Fake News**
  - Selected for its detailed inclusion of source information, crucial for verifying authenticity.
- Special thanks to the creators and contributors of this dataset for enabling research in combating misinformation.
  
A huge **thank you** to **ruchi798** for providing the dataset that made this project possible! 🌟 The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification/data). Your contribution is greatly appreciated! 🙌

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

Throughout the development of this project, I drew inspiration from several community contributions that tackled fake news classification from different angles. One particularly valuable resource was the Kaggle notebook by **Vishesh Bansal**, titled [*EDA and Modelling on News Dataset (99% accuracy)*](https://www.kaggle.com/code/bansalvishesh/eda-and-modelling-on-news-dataset-99-accuracy).

This notebook provides a thorough exploratory data analysis of news content and experiments with text preprocessing techniques such as **TF-IDF** and **word embeddings** using TensorFlow. I am grateful for the educational value of such community-driven contributions, which not only accelerate individual learning but also foster collaborative research and shared growth within the field.

## Visitors Count

<img width="auto" src="https://profile-counter.glitch.me/fake_news_classifier/count.svg" />

## Please Share & Star the repository to keep me motivated.
<a href = "https://github.com/sergio11/fake_news_classifier/stargazers">
   <img src = "https://img.shields.io/github/stars/sergio11/fake_news_classifier" />
</a>

## License ⚖️

This project is licensed under the MIT License, an open-source software license that allows developers to freely use, copy, modify, and distribute the software. 🛠️ This includes use in both personal and commercial projects, with the only requirement being that the original copyright notice is retained. 📄

Please note the following limitations:

- The software is provided "as is", without any warranties, express or implied. 🚫🛡️
- If you distribute the software, whether in original or modified form, you must include the original copyright notice and license. 📑
- The license allows for commercial use, but you cannot claim ownership over the software itself. 🏷️

The goal of this license is to maximize freedom for developers while maintaining recognition for the original creators.

```
MIT License

Copyright (c) 2024 Dream software - Sergio Sánchez 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``
