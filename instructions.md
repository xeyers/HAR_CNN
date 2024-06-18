# Handy guide how to run this project
## Disclaimer
This is a brief instruction covering how to run code on this repository. It covers basics without diving too deep into the details.

If you want to use different/less/more activities just make sure to adjust any cell that are related to it.
## Data handling
All files to run are in Jupyter Notebook so you will need to either install it or use their website.
1. Run cells with imports and function definitions, just run them, there is nothing to change.
2. Next cell is important
```py
file = open("YOUR_DATASET.txt")  # Dataset
lines = file.readlines()

processedData = []  # Dataset after processing, it is necessary in order to exclude incorrectly formatted data

for i, line in enumerate(lines):
    try:
        line = line.split(";")
        last = line[2].split("\n")[0] # bylo 3
        last = last.strip()
        if last == "":
            break
        temp = [line[0], line[1], last] # temp = [line[0], line[1], line[2], last]
        processedData.append(temp)
    except:
        print("Error in line: ", i)

columns = ["activity", "y", "z"] # columns = ["activity", "x", "y", "z"]
data = pd.DataFrame(data = processedData, columns = columns)
```
Here you have "dataset" variable. Make sure to put path to your dataset here (make sure it has the right amount of axes). It doesn't have to be txt, can be csv as well, just make sure each field is separated with a semicolon (;) and each line ends with Enter. My dataset was in format "activity;y;z" but yours might be different. Make sure last variable is set to the last column of your line and temp variable to however you want to arrange your dataset. Make sure columns variable is set accordingly. If you don't want to spend 2 hours editing whole code make sure it follows one of those 2 formats:
```
activity;y_axis;z_axis
```
or
```
activity;x_axis;y_axis;z_axis
```

 Next important thing is the line
```py
columns = ["activity", "y", "z"] # columns = ["activity", "x", "y", "z"]
```
The commented code is version for 3 axis, if you want to use just two as recommended don't change anything.
3. Keep on running cells in order they are. For now all cells just handle making sure that data fed to the model is data it can understand. You will occasionally run into cells like this.
```py
#data["x"] = data["x"].astype(float)
data["y"] = data["y"].astype(float)
data["z"] = data["z"].astype(float)
```
and (note that cell below has X but it is not X axis)
```py
X = balanced_data[["y","z"]] # X = balanced_data[["x","y","z"]]
y = balanced_data["label"]
```
In cells like this it is up to you to choose if you want to use all 3 axes or not. But make sure you don't mix it because it won't work, use either 3 (not recommended) or 2 (recommended).
4. This step is very important. You will eventually run into this cell
```py
X_train.shape, X_test.shape
```
and it will give some output, below I put example one:
```
((137, 80, 2), (35, 80, 2))
```
Each tuple has 3 numbers, the one that you are interested in is first number in each tuple. 2 cell later you encounter this cell
```py
# 2 axes
X_train = X_train.reshape(137,80,2,1) 
X_test = X_test.reshape(35,80,2,1) 

# 3 axes
# X_train = X_train.reshape(64,80,3,1)
# X_test = X_test.reshape(16,80,3,1)
```
As you can see those numbers are here as well on first position. Depending on your data set you will have to update them manually according to the output from the cell:
```py
X_train.shape, X_test.shape
```
## Running the model
### Option 1: Train the model
Simply run the cell
```py
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input

# Define the model
model = Sequential()
model.add(Input(shape=(80, 2, 1)))  # Specify input shape using Input layer
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same'))  # First layer with 'same' padding
model.add(Dropout(0.1))  # 10% of neurons will be dropped randomly

model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'))  # Second layer with 'same' padding
model.add(Dropout(0.2))  # 20% of neurons will be dropped randomly

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 50% of neurons will be dropped randomly

model.add(Dense(6, activation='softmax'))  # 6 because we have 6 classes

model.compile(optimizer=Adam(learning_rate=0.003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=250, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save('YOUR_MODEL_NAME.keras')
```
Your model will be trained, saved and ready to use! To check performance run the next cell that will draw error matrices for you

```py
truth = ["Jogging", "Sitting", "Standing", "Stairs", "Walking"]
prediction = ["Jogging", "Sitting", "Standing", "Stairs", "Walking"]

y_pred = np.argmax(model.predict(X_train), axis=-1)
mat = confusion_matrix(y_train, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7, 7))
plt.show()  # You also need to add this line to display the plot

print_confusion_matrix(mat, label.classes_)
```
You will get 2 matrices, one normal one and one heatmap looking one.
### Option 2: Use existing model
You still have to run all data related cells.
I might have had accidentaly left some not working cells at the end, so if you see them you can deleted them.

Add this cell after running all data related cells:
```py
from keras.src.saving import load_model

# Load the model from the file
model = load_model('YOUR_MODEL.keras')

truth = ["Jogging", "Sitting", "Standing", "Stairs", "Walking"]
prediction = ["Jogging", "Sitting", "Standing", "Stairs", "Walking"]

y_pred = np.argmax(model.predict(X_train), axis=-1)
mat = confusion_matrix(y_train, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7, 7))
plt.show()  
print_confusion_matrix(mat, label.classes_)
```
