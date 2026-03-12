import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle as pkl

df = pd.read_csv(r"C:\Users\aksha\Documents\pyth\iris.data",names=["sepal length in cm","sepal width in cm","petal length in cm","petal width in cm", "species"])
df

le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print("Accuracy:", accuracy)

with open ('mm1.pkl','wb') as file:
    pkl.dump(gb,file)
