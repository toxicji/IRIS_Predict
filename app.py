from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

app = FastAPI()
templates = Jinja2Templates(directory="templates")




iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(64, input_dim=4, activation = 'relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=2, validation_data=(X_test, y_test))


@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    def predict_flower_species(sepal_length, sepal_width, petal_length, petal_width):
        
        input_data = [sepal_length, sepal_width, petal_length, petal_width]
        
        predictions = model.predict(np.array(input_data).reshape(1, -1))
        predicted_class = np.argmax(predictions)

        flower_species = iris.target_names[predicted_class]

        return flower_species
    predicted_species = predict_flower_species(sepal_length, sepal_width, petal_length, petal_width)

    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": f"Предсказанное наименование цветка: {predicted_species}"})