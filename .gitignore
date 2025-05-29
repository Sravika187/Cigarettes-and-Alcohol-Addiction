import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1. Simulated dataset for addiction detection
data = {
    'text': [
        'I smoke a pack a day, can’t quit.',
        'I only drink on weekends.',
        'Been trying to quit smoking but it’s really hard.',
        'I don’t drink or smoke.',
        'Alcohol has taken over my life.',
        'I enjoy a glass of wine occasionally.',
        'Nicotine cravings are unbearable.',
        'My liver is suffering from years of alcohol use.',
        'I’ve never touched a cigarette in my life.',
        'I quit drinking two years ago.'
    ],
    'label': [
        'CIGARETTE', 'ALCOHOL', 'CIGARETTE', 'NONE',
        'ALCOHOL', 'NONE', 'CIGARETTE', 'ALCOHOL',
        'NONE', 'ALCOHOL'
    ]
}

df = pd.DataFrame(data)

# 2. Preprocessing
X = df['text']
y = df['label']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Model creation
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100)

# 6. Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('lr', log_reg), ('rf', random_forest)
], voting='hard')

# 7. Hyperparameter Tuning for Logistic Regression
param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 200, 300]}
grid_search = GridSearchCV(log_reg, param_grid, cv=3)
grid_search.fit(X_train_tfidf, y_train)

# 8. Use best parameters
best_log_reg = grid_search.best_estimator_

# 9. Train the ensemble
ensemble_model.fit(X_train_tfidf, y_train)

# 10. Evaluate
y_pred = ensemble_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Ensemble Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['CIGARETTE', 'ALCOHOL', 'NONE'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['CIGARETTE', 'ALCOHOL', 'NONE'],
            yticklabels=['CIGARETTE', 'ALCOHOL', 'NONE'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 12. Metrics Bar Plot
metrics = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(metrics).transpose()
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 5))
plt.title('Precision, Recall, F1-Score Comparison')
plt.ylabel('Score')
plt.xlabel('Class')
plt.xticks(rotation=45)
plt.show()

# 13. Save the model and vectorizer
joblib.dump(ensemble_model, 'addiction_model.pkl')
joblib.dump(vectorizer, 'addiction_vectorizer.pkl')

# 14. Predict user input
while True:
    user_input = input("\nEnter a text related to substance use (type 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        break
    input_vec = vectorizer.transform([user_input])
    prediction = ensemble_model.predict(input_vec)[0]
    print(f"🧠 Prediction: This text indicates {prediction.upper()} addiction category.")
