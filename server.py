import canvas as canvas
import tkinter as tk
from tensorflow import keras
import numpy as np
from PIL import ImageTk, Image
import csv
import os
import spacy
from spacy.matcher import PhraseMatcher
from tkinter import filedialog
from tkinter import Tk, Label, messagebox
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

remember_keys=[]
memory_key=[]
displayed_content = []

history=[]

# Define the path to the image dataset folder
dataset_folder = "Images_dataset"
# Define the path to the annotations CSV file
csv_file = "clothes_data.csv"

# Load the annotations from the CSV file
annotations = []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        annotations.append(row)

# Load the language model
nlp = spacy.load('en_core_web_sm')

def extract_column_values(csv_file, columns):
    extracted_values = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            values = [row[column] for column in columns]
            extracted_values.extend(values)
    return extracted_values

# Specify the CSV file path and columns to extract
columns_to_extract = ['color', 'type', 'gender', 'pattern1', 'pattern2',
                      'components1', 'components2','components3']

# Extract the content of the specified columns
extracted_data = extract_column_values(csv_file, columns_to_extract)

# Prepare clothing attributes data
clothing_attributes = extracted_data

# Initialize the PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(attribute.lower()) for attribute in clothing_attributes]
matcher.add("ClothingAttributes", None, *patterns)

# Function to search and display clothing images based on user input
def search_clothing():

    global search_button_clicked
    search_button_clicked = True
    global user_sentence
    user_sentence=input_text.get("1.0", "end-1c").strip()
    user_input = input_text.get("1.0", "end-1c").strip().lower()
    if not user_input:
        # Detect if the user has input useful information
        print("The user did not enter correct and valid information")
        messagebox.showinfo("Warning", "The information you entered is empty\n Please re-enter it")
        return

    # Use NLP to analyze the user's language
    doc = nlp(user_input)
    global keywords

    # Extract keywords from the user's description
    keywords = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    print("User input keywords:", keywords)

    if len(keywords) == 0:
        # Detect if the user has input useful information
        print("Please enter correct and valid information")
        messagebox.showinfo("Warning", "Please enter correct, valid, and detailed information")
        return

    clothe_type = ["Baby Onesie", "Coat", "Dress", "Dungarees", "Hoodie", "Jeans", "Pants", "Shirt", "Shorts", "Skirt",
                   "Strap dress", "Suit set", "Sweater", "T-shirt", "Vest", "Wedding Dress"]
    clothe_color = ['White', 'Brown', 'Red', 'Black', 'Green', 'Beige', 'Orange', 'Blue', 'Grey', 'Purple', 'Yellow',
                    'Pink']
    clothe_gender = ['Female', 'Unisex', 'Male']
    clothe_pattern = ['Pure color', 'Embroidery', 'Fruit', 'Lace', 'Abstract', 'Alphabet', 'Plaid', 'Floral', 'Cartoon',
                      'Geometric', 'Animal', 'Stripes', 'Leopard print', 'NA', 'Polka dots']
    clothe_component = ['Elastic', 'High collar', 'Collar', 'Sleeve', 'Drawstring', 'Zipper', 'Pleat', 'Tie', 'Button',
                        'NA', 'Hem', 'Hood', 'Belt', 'Ruffle', 'Pocket', 'Bow']

    # Merge the list of clothes into one big text and convert to lowercase
    clothe_corpus = [word.lower() for word in
                     clothe_type + clothe_color + clothe_gender + clothe_pattern + clothe_component]

    # Calculate TF-IDF features using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    clothe_tfidf = vectorizer.fit_transform(clothe_corpus)

    # Calculate the TF-IDF features of the user input text
    user_input_tfidf = vectorizer.transform([user_input.lower()])

    # Calculate the similarity between the user input text and the clothe list
    similarities = cosine_similarity(user_input_tfidf, clothe_tfidf)[0]
    # Define the similarity threshold
    similarity_threshold = 0.45
    keywords = []
    clothe_dict={}
    # Output matching result
    for i, similarity in enumerate(similarities):
        if similarity > similarity_threshold:
            # Replace with the corresponding clothes list as needed
            matched_clothe = clothe_corpus[i]
            print(f"The text entered by the user and '{matched_clothe}' the similarity is: {similarity}")
            clothe_dict[matched_clothe] = similarity
            keywords.append(matched_clothe)

    clothe_type_lower = [item.lower() for item in clothe_type]
    filtered_dict = {key: value for key, value in clothe_dict.items() if key in clothe_type_lower}
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
    first_key = next(iter(sorted_dict))
    keys_list = list(sorted_dict.keys())
    keys_list.remove(first_key)
    keywords = [x for x in keywords if x not in keys_list]

    # Search for clothing images matching the matched attribute
    image_matches = []
    remember()
    keywords.extend(memory_key)
    print("finally keywords:",keywords)
    for annotation in annotations:
        predicted_class = annotation['type']
        dominant_colors = annotation['color']
        gender = annotation['gender']
        pattern1 = annotation['pattern1']
        pattern2 = annotation['pattern2']
        components1 = annotation['components1']
        components2 = annotation['components2']
        components3 = annotation['components3']
        colthes_columns = [predicted_class.lower(), dominant_colors.lower(), gender.lower(), pattern1.lower(), pattern2.lower(),components1.lower(), components2.lower(),components3.lower()]
        if len(keywords) == 2 :
            if any(keyword in predicted_class.lower() for keyword in keywords) and \
                    any(keyword in dominant_colors.lower() for keyword in keywords):
                image_matches.append(annotation)

        if len(keywords) == 1:
            if any(keyword in predicted_class.lower() for keyword in keywords) or \
                    any(keyword in dominant_colors.lower() for keyword in keywords):
                image_matches.append(annotation)

        if all(keyword.lower() in colthes_columns for keyword in keywords) :
            image_matches.append(annotation)

    # Display the clothing images
    display_content(image_matches)

def remember():
    global remember_keys
    remember_keys.extend(keywords)
    print('remember_keys:',remember_keys)
    mid_key=Counter(remember_keys)
    high_to_low =[item for item, count in mid_key.most_common()]
    print('high_to_low:',high_to_low)
    for i in keywords:
        if i in high_to_low:
            high_to_low.remove(i)
    print('high_to_low:',high_to_low)

    global memory_key

    cloth_types=['baby onesie','coat','dress','dungarees','hoodie','jeans','pants','shirt','shorts','skirt','strap dress','suit set','sweater','t-shirt','vest','wedding dress']
    if any(cloth_type in high_to_low for cloth_type in cloth_types) and \
            any(cloth_type in keywords for cloth_type in cloth_types):
        for elemt in cloth_types:
            if elemt in high_to_low:
                high_to_low.remove(elemt)
    if len(high_to_low)==0:
        memory_key = []
    else:
        memory_key = [high_to_low[0]]
    print(memory_key)
    print("------------------")

def display_content(matches):
    global displayed_content
    if len(matches) == 0:
        # No matching images found
        print("No matching images found")
        messagebox.showinfo("No Matches", "Sorry, there are no matching clothing images.")
        return
    if len(matches)<=6:
        top_matches = matches[:3]  # Get the top three matches
    else:
        top_matches = matches[:6:2]
    # Create a new Label and image
    content_frame = tk.Frame(inner_frame, bg="#EAEAEA")
    content_frame.grid(sticky="w", padx=10, pady=(10, 0))

    text_label = tk.Label(content_frame, text=user_sentence)
    text_label.pack()

    image_frame = tk.Frame(content_frame, bg="#EAEAEA")
    image_frame.pack()

    for image_name in top_matches:
        image_path = os.path.join(dataset_folder, image_name['image'])
        image = Image.open(image_path)
        image = image.resize((100, 100))
        photo = ImageTk.PhotoImage(image)
        img_label = tk.Label(image_frame, image=photo)
        img_label.image = photo
        img_label.pack(side=tk.LEFT, padx=5)

    # Update the displayed content list
    displayed_content.append(content_frame)
    # Update the scroll range
    scroll_region()


def scroll_region():
    canvas.configure(scrollregion=canvas.bbox("all"))

def on_mouse_wheel(event):
    canvas.yview_scroll(-1 * (event.delta // 120), "units")

def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

def put_search_clothing(keywords):
    # Define the path to the annotations CSV file
    csv_file = "clothes_data.csv"
    another_annotations = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            another_annotations.append(row)
    image_matches = []
    for annotation in another_annotations:
        predicted_class = annotation['type']
        if keywords.lower() in predicted_class.lower():
            image_matches.append(annotation)
    # Display the clothing images
    put_display_images(image_matches)


def put_display_images(matches):
    # Clear any existing images
    global displayed_content
    if len(matches) == 0:
        # No matching images found
        print("No matching images found")
        messagebox.showinfo("No Matches", "Sorry, there are no matching clothing images.")
        return
    if len(matches) <= 6:
        top_matches = matches[:3]  # Get the top three matches
    else:
        top_matches = matches[:6:2]
    # Create a new Label and image
    content_frame = tk.Frame(inner_frame, bg="#EAEAEA")
    content_frame.grid(sticky="w", padx=10, pady=(10, 0))

    text_label = tk.Label(content_frame, text=predicted_class_name)
    text_label.pack()

    image_frame = tk.Frame(content_frame, bg="#EAEAEA")
    image_frame.pack()

    for image_name in top_matches:
        image_path = os.path.join(dataset_folder, image_name['image'])
        image = Image.open(image_path)
        image = image.resize((100, 100))
        photo = ImageTk.PhotoImage(image)
        img_label = tk.Label(image_frame, image=photo)
        img_label.image = photo
        img_label.pack(side=tk.LEFT, padx=5)

    # Update the displayed content list
    displayed_content.append(content_frame)
    # Update the scroll range
    scroll_region()


def select_image():

    # Open file selection dialog
    select_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    # Check if file is selected
    global select_file
    select_file=select_file_path
    if select_file_path:
        try:
            # Create an Image object
            image = Image.open(select_file_path)
            # Resize image to fit label
            image = image.resize((200, 200))
            # Create a PhotoImage object to display the image on the label
            photo = ImageTk.PhotoImage(image)
            # Get the selected image path in the background
            print("Selected image path:", select_file_path)
        except IOError:
            print("Can not open image file:", select_file_path)


def analysis_pho():
    class_names = ['Baby Onesie', 'Coat', 'Dress', 'Dungarees', 'Hoodie', 'Jeans', 'Pants', 'Shirt',
                   'Shorts', 'Skirt', 'Strap dress', 'Suit set', 'Sweater', 'T-shirt', 'Vest',
                   'Wedding Dress']
    num_classes = 16
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Load previously saved model weights
    model.load_weights('model_weights.h5')

    # Use the model to make predictions
    image_path = select_file  # Specify the image path to be predicted
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    global predicted_class_name
    predicted_class_name = class_names[predicted_class_index]

    print(f"Predicted class: {predicted_class_name}")
    # Specify the folder path where the picture to be predicted is located
    test_data_dir = select_file
    put_search_clothing(predicted_class_name)

# Create the GUI window
window = tk.Tk()
window.title("Recommendation System")
window.configure(bg='white')
window.geometry('1150x700')

# Create a scrollbar
scrollbar = tk.Scrollbar(window)
scrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)

# Create a canvas
canvas = tk.Canvas(window, yscrollcommand=scrollbar.set)
canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

# Configure the scrollbar to work with the canvas
scrollbar.configure(command=canvas.yview)

# Adjust row and column weights to make the canvas expand
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

scrollbar.config(command=canvas.yview)

content_frame = tk.Frame(canvas)  # put the content in a frame
canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)

# Bind the update_scroll_region function to the content frame's size change event
content_frame.bind("<Configure>", update_scroll_region)

# Create title label
title_label = tk.Label(content_frame, text="Clothing Recommendation System", font=('Arial', 16, 'bold'), pady=10, anchor='center')
title_label.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky='n')

# Create description label
description_label = tk.Label(content_frame, text="Enter your clothing description:", font=('Arial', 12))
description_label.grid(row=1, column=0, padx=10)

putphoto_label = tk.Label(content_frame, text="or please input the image:", font=('Arial', 12))
putphoto_label.grid(row=2, column=2)

# Create the input text box for user description
input_text = tk.Text(content_frame, height=10, width=50, bg='lightgray', font=('Arial', 12))
input_text.grid(row=1, column=1, padx=10, pady=10, sticky='w')

# Create the "Search" button
search_button = tk.Button(content_frame, text="Search", command=search_clothing, bg='blue', fg='white', font=('Arial', 14, 'bold'))
search_button.grid(row=2, column=1, padx=10, pady=10)

put_button = tk.Button(content_frame, text="Select", bg='green',fg='white',font=('Arial', 14, 'bold'),command=select_image)
put_button.grid(row=2, column=3, padx=10, pady=10)

analysis_button=tk.Button(content_frame, text="Analysis", bg='green',fg='white',font=('Arial', 14, 'bold'),command=analysis_pho)
analysis_button.grid(row=2, column=4, padx=10, pady=10)

content_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox(tk.ALL))
# Use the mouse wheel event
canvas.bind("<MouseWheel>", on_mouse_wheel)
update_scroll_region(None)

# Create a new frame for the small canvas
canvas_frame = tk.Frame(content_frame, bg="#EAEAEA")
canvas_frame.grid(row=4, column=1, columnspan=3, padx=10, pady=10)

letter_label = tk.Label(canvas_frame, text="Chat Photos:", font=('Arial', 12, 'bold'), bg="white")
letter_label.pack(side=tk.TOP, pady=10)

# Create the small canvas
small_canvas = tk.Canvas(content_frame, bg="#CCCCCC", yscrollcommand=scrollbar.set)
small_canvas.grid(row=5, column=1, columnspan=3, padx=10, pady=10, sticky='nsew')

# Create a vertical scrollbar for the small canvas
scrollbar_small = tk.Scrollbar(content_frame, orient=tk.VERTICAL, command=small_canvas.yview)
scrollbar_small.grid(row=5, column=4, sticky=tk.N+tk.S)

# Configure the small canvas to work with the new scrollbar
small_canvas.configure(yscrollcommand=scrollbar_small.set)

# Create the inner frame for the small canvas
inner_frame = tk.Frame(small_canvas, bg="#CCCCCC")

# Bind the inner frame to the small canvas
inner_frame.bind("<Configure>", lambda event: small_canvas.configure(scrollregion=small_canvas.bbox("all")))

# Create the window for the inner frame in the small canvas
small_canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

# Update the size and scroll region of the canvas frame
canvas_frame.update_idletasks()
small_canvas.config(scrollregion=small_canvas.bbox(tk.ALL))

# Run the GUI window
window.mainloop()



