# Restaurant Reviews (NLP model) 

### Table of Contents
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Run](#run)
- [Deployment on Render](#deployment-on-render)
- [Directory Tree](#directory-tree)
- [To Do](#to-do)
- [Bug / Feature Request](#bug--feature-request)
- [Technologies Used](#technologies-used)
- [Team](#team)
- [License](#license)
- [Credits](#credits)

---

## Demo
This project analyzes customer reviews to determine if the food in a restaurant was liked or disliked based on textual feedback. 
**Link to Demo:** [Steam Review Sentiment Analysis(#) ]

## Restaurant Reviews (NLP model) 

![Restaurant Review Sentiment Analysis](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzsUUW4DARyWhxiWME9dypkC7WMkcXKG-Xsw&s)


---

## Overview
The Restaurant Reviews Sentiment Analysis project leverages Natural Language Processing (NLP) techniques to analyze customer reviews and classify them as either positive (liked) or negative (disliked). The goal is to provide restaurant owners, managers, and businesses with insights into customer satisfaction based on text feedback.

Key features:

- Preprocessing of review text data (cleaning, tokenization, and stemming).
- Sentiment classification using machine learning models.
- Interactive web application for real-time predictions.
---

## Motivation
Sentiment analysis of restaurant reviews allows businesses to:

- Gain insights into customer opinions.
- Monitor public perception of restaurant services.
- Detect areas for improvement based on customer sentiment.
- This project showcases the practical application of NLP and machine learning for analyzing customer feedback.
---

## Technical Aspect
### Training Machine Learning Models:

Data Collection:

- Reviews are collected from various sources, such as publicly available restaurant review datasets.
Preprocessing:

- Remove unnecessary characters like punctuation, special symbols, etc.
- Tokenize the text, remove stop-words, and apply stemming to reduce words to their base forms.
Feature Extraction:

- Convert text data into numerical representations using techniques like TF-IDF or Word2Vec.
Model Training:

- Various models such as Logistic Regression, Support Vector Machines (SVM), or BERT are employed for sentiment classification.
Hyperparameter tuning is done to improve the accuracy of the model.
Model Evaluation:

- Evaluate models using metrics like accuracy, precision, recall, and F1 score.
Web App Development:

- A Flask-based web application is used to process user-inputted reviews and display sentiment predictions.
---

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

# To clone the repository

```bash

gh repo clone Creator-Turbo/Restaurant-Reviews

```
# Install dependencies: (all lib)
```bash
pip install -r requirements.txt
```



## Run
To train the Machine leaning models:
 To run the Flask web app locally
```bash
python app.py

```
# Deployment on Render

## To deploy the Flask web app on Render:
Deployment on Render

- To deploy the web app on Render:

- Push your code to GitHub.

- Log in to Render and create a new web service.

- Connect the GitHub repository.

- Configure environment variables (if any).

- Deploy and access your app live.


## Directory Tree 
```
.
├── data
│   └── (files inside data directory)
├── model
│   └── (files inside model directory)
├── notebook
│   └── (files inside notebook directory)
├── venv
│   └── (virtual environment files)
├── webapp
│   └── (files inside webapp directory)
├── .gitignore
├── README.md
└── requirements.txt


```

## To Do

- Expand dataset to improve model robustness.

- Experiment with advanced models like BERT or GPT-based sentiment classifiers.

- Add sentiment trend visualization to the web app.

- Automate data collection using the Twitter API.






## Bug / Feature Request
If you encounter any bugs or want to request a new feature, please open an issue on GitHub. We welcome contributions!




## Technologies Used
- Python 3.10: Programming language used for implementing the model.

- scikit-learn: For machine learning algorithms and model evaluation.
- Flask: For developing the web application.
- Render: For deploying the web application.
- pandas: For data manipulation and preprocessing.
- numpy: For numerical computations.
- matplotlib: For visualizations.
- nltk: For natural language processing tasks (tokenization, stop-word removal, stemming).




![](https://forthebadge.com/images/badges/made-with-python.svg)


[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png" width=170>](https://pandas.pydata.org/docs/)
[<img target="_blank" src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*RWkQ0Fziw792xa0S" width=170>](https://pandas.pydata.org/docs/)
  [<img target="_blank" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDzf1RMK1iHKjAswDiqbFB8f3by6mLO89eir-Q4LJioPuq9yOrhvpw2d3Ms1u8NLlzsMQ&usqp=CAU" width=280>](https://matplotlib.org/stable/index.html) 
 [<img target="_blank" src="https://icon2.cleanpng.com/20180829/okc/kisspng-flask-python-web-framework-representational-state-flask-stickker-1713946755581.webp" width=170>](https://flask.palletsprojects.com/en/stable/) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width=200>](https://aws.amazon.com/s3/) 
 [<img target="_blank" src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" width=200>](https://seaborn.pydata.org/generated/seaborn.objects.Plot.html) 







## Team
This project was developed by:
[![Bablu kumar pandey](https://github.com/Creator-Turbo/images-/blob/main/resized_image.png?raw=true)](ressume_link) |
-|


**Bablu Kumar Pandey**


- [GitHub](https://github.com/Creator-Turbo)  
- [LinkedIn](https://www.linkedin.com/in/bablu-kumar-pandey-313764286/)
* **Personal Website**: [My Portfolio](https://creator-turbo.github.io/Creator-Turbo-Portfolio-website/)

## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this software under the terms of the MIT License. For more details, see the [LICENSE](LICENSE) file included in this repository.


## Credits

Special thanks to the contributors of the scikit-learn library for their fantastic machine learning tools.