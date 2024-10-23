import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
data = {
    'text': [
        "The doctor was very helpful and attentive during my visit.",
        "I had a terrible experience with the hospital staff.",
        "The treatment was effective and I feel much better now.",
        "The waiting time was too long, and the nurses were rude.",
        "Great service! The team was very professional.",
        "I wouldn't recommend this clinic due to poor service."
    ],
    'sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative']
}
df = pd.DataFrame(data)
df['label'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
idx = 0  # Change this index to analyze different samples
text_instance = X_test.iloc[idx]
exp = explainer.explain_instance(text_instance, model.predict_proba, num_features=5)
exp.show_in_notebook(text=True)
pred_prob = model.predict_proba([text_instance])[0]
print(f"Prediction probabilities for the instance '{text_instance}':")
print(f"Negative: {pred_prob[0]:.4f}, Positive: {pred_prob[1]:.4f}")
fig = exp.as_pyplot_figure()
plt.title('LIME Explanation for Healthcare Text Classification')
plt.show()