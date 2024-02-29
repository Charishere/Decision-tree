import os
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from tkinter import messagebox
from tkinter import font 


clf = None
y = None
X_train = None
y_train = None
X_test = None
y_test = None
y_pred = None
data_feature = []
accuracy_value = None
trees = []
num_trees = None

#gui window
import tkinter as tk
from tkinter import *
window = tk.Tk()
window.title("Desicion Tree Model")
window.resizable(width=False, height=False)

frame = tk.Frame(relief=tk.FLAT, border=5)
frame.grid(row=0, column=0)


#data line
data_line = tk.Frame(frame)
data_line.grid(row=0, column=0, sticky="w", pady=3)

##select file
entry = tk.Entry(master=data_line, width=30)
entry.grid(row=0, column=0,sticky="w")

def get_path(self, event):
    self.filepath = event.GetString()

from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
def select_file():
    filetypes = (
        ('csv files', '*.csv'),
        ('xlsx files', '*.xlsx'),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(
        initialdir='/',
        filetypes=filetypes
    )
    entry.delete(0, tk.END)
    entry.insert(0, filename)

select_but = tk.Button(master=data_line, text="Select", relief=tk.RAISED, command=select_file)
select_but.grid(row=0,column=1, sticky="w", padx=4)

##load file
from sklearn.preprocessing import LabelEncoder
def load_data():
    global X_train, y_train, X_test, y_test, data_feature, y
    filepath = entry.get()
    print("File path:", filepath)
    try:        
        data = pd.read_csv(filepath)
        data_feature = data.columns.values.tolist()[:len(data.columns)-1]
        col_names =  data.columns.values.tolist()[:len(data.columns)]
        X = data[data_feature]
        y = data[data.columns.values.tolist()[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        label_encoder = LabelEncoder()
        for col in X_train.columns:
            if X_train[col].dtype == "object":
                X_train[col] = label_encoder.fit_transform(X_train[col])
        for col in X_test.columns:
            if X_test[col].dtype == "object":
                X_test[col] = label_encoder.fit_transform(X_test[col])
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
        messagebox.showinfo("Info", "Data loaded successfully")
    except FileNotFoundError:
        messagebox.showerror("Error", "File not found")
    except pd.errors.EmptyDataError:
        messagebox.showerror("Error", "File is empty")
    except Exception as e:
        messagebox.showerror("Error", "Failed to load data")

load_but = tk.Button(master=data_line, text="Load", command=load_data)
load_but.grid(row=0,column=2,sticky="w", padx=0)


#train line
train_line = tk.Frame(frame)
train_line.grid(row=1, column=0, sticky="w", pady=3)

label_alg = tk.Label(master=train_line, text="Choose the algorithm")
label_alg.grid(row=1, column=0, sticky="w")

##tree list
tree_line = tk.Frame(frame)
tree_line.grid(row=2, column=0, sticky="w", pady=3)

label_tre = tk.Label(master=tree_line, text="How many trees")
label_tre.grid(row=2, column=0, sticky="w")
tre_ent = tk.Entry(master=tree_line, width=3)
tre_ent.grid(row=2, column=1,sticky="w",padx=4)

def how_mang_trees():
    global accuracy_value, num_trees
    try:
        trees = None
        accuracy_value = None
        num_trees = int(tre_ent.get())        
        if (num_trees>10):
            messagebox.showerror("Error", "Invalid number. Please enter a number less than or equal to 10.")
        else:            
            return num_trees
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter a valid number.")

def select_input(value):
    if value is not None:
        selected_lab.config(text=f"Selected: {value}")

def entre(event):
    select_input(how_mang_trees())
tre_ent.bind("<Return>", entre)
selected_lab = tk.Label(master=tree_line, text="Selected: ")
selected_lab.grid(row=2, column=2, padx=3, ipady=4) 

##toggle list
def toggle_list():
    if list_frame.winfo_ismapped():
        list_frame.grid_forget()
    else:
        list_frame.grid(row=1,column=0, sticky="w",padx=94, ipadx=8,pady=0)

def select_item(value):
    global clf
    selected_label.config(text=f"Selected: {value}")
    if value == "ID3":
        clf = DecisionTreeClassifier(criterion="entropy")
    else:
        clf = DecisionTreeClassifier()
    toggle_list()

items = ["ID3", "CART"]
toggle_but = tk.Button(master=train_line, text="Choose", command=toggle_list)
toggle_but.grid(row=1,column=1, sticky="w",padx=5, ipadx=0)

list_frame = tk.Frame(frame)
selected_item = tk.StringVar(frame)
selected_item.set(items[0]) # Default selected item

option_menu = tk.OptionMenu(list_frame, selected_item, *items, command=select_item)
option_menu.grid(pady=4)
selected_label = tk.Label(master=train_line, text="Selected: ")
selected_label.grid(row=1, column=2, padx=3, ipady=4) 
list_frame.grid_forget()


#image
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from tkinter import *
from PIL import ImageTk, Image
def generate_and_display_tree():
    global data_feature, trees, num_trees, y
    n = num_trees
    images = []
    originals = []
    index = 0
    class_names = list([str(value) for value in y.unique()])   
        
    if len(trees) != 0:
        for tree in trees:
            dot_data = StringIO()
            print(tree)
            export_graphviz(tree, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True, feature_names=data_feature, class_names=class_names
                            )
        

            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(f'tree {index}.png')
            img = Image.open(f'tree {index}.png') 
            originals.append(img)       
            img_p = img.resize((1000, 500), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_p)
            panel.configure(image=img_tk)
            panel.image = img_tk
            panel.original_image = img
            index += 1
            images.append(img_tk)
            print(images)
        viewer = TreeViewer(images, originals)
        panel.bind("<Button-1>", lambda event: viewer.show())
    else:
        messagebox.showerror("Error","Please train first.")

class TreeViewer:
    def __init__(self, images, original_image):
        self.images = images
        self.index = 0
        self.viewer_window = None
        self.original_image = original_image

    def show(self):
        if not self.viewer_window:
            self.viewer_window = Toplevel()
            self.viewer_window.title("Tree Viewer")
            
            self.image_label = Label(self.viewer_window)
            self.image_label.pack()

            btn_prev = Button(self.viewer_window, text="<", command=self.show_prev)
            btn_next = Button(self.viewer_window, text=">", command=self.show_next)
            btn_prev.pack(side=LEFT)
            btn_next.pack(side=RIGHT)

            self.image_label.bind("<Button-1>", lambda event: self.open_default_viewer())
            
        self.show_image()

    def show_image(self):
        if self.viewer_window and self.image_label.winfo_exists():
            img = self.images[self.index]
            title = f"Tree {self.index + 1}"
            self.image_label.configure(image=img, text=title, compound='bottom')
            self.image_label.image = img
            self.image_label.title = title
        else:
            messagebox.showerror("Error","Please replot")

    def show_next(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_image()

    def show_prev(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()
    
    def open_default_viewer(self):
        if self.index < len(self.original_image):
            original_image = self.original_image[self.index]
            original_image.show()
        


image_panel_frame = Frame(frame)
image_panel_frame.grid(row=3, column=0, pady=10, sticky="w")

img = ImageTk.PhotoImage(Image.new("RGB", (1000, 500), "white"))
panel = Label(image_panel_frame, image=img)
panel.image = img
panel.grid(row=0, column=0)

panel.zoom_level = 1.0



#function line
model_buttons = tk.Frame(frame)
model_buttons.grid(row=4, column=0, sticky="w", pady=10)

def train():
    global clf, X_train, y_train, accuracy_value, trees, num_trees
    accuracy_value = None
    if clf is None:
        messagebox.showerror("Error", "Please choose a valid algorithm.")
    elif num_trees is None:
        messagebox.showerror("Error", "Invalid input. Please enter a valid number.")
    elif clf is not None and X_train is not None and y_train is not None:
        print(clf)
        trees = []
        if num_trees == 1:
            clf = clf.fit(X_train, y_train)
            trees.append(clf)
        else:
            for _ in range(num_trees):
                current_clf = clf.__class__()
                current_clf = current_clf.fit(X_train, y_train)
                trees.append(current_clf)
            return trees
    else:
        messagebox.showerror("Error",f"Training failed\nNo data found")
train_but = tk.Button(master=model_buttons, text="Train", command=train)

plot_but = tk.Button(master=model_buttons,command=generate_and_display_tree, text="Plot")

#accuracy
def accu():
    global clf, X_test, y_test, accuracy_value, trees, num_trees
    if accuracy_value is not None:
        return accuracy_value
    accuracies = []
    n = num_trees
    if clf is not None:
        try:
            if X_test is not None and y_test is not None:
                if n == 1:
                    y_pred = clf.predict(X_test)
                    accu = metrics.accuracy_score(y_test, y_pred)
                    accuracy_value = accu                    
                    return accuracy_value
                else:
                    for tree in trees:
                        y_pred = tree.predict(X_test)
                        accu = metrics.accuracy_score(y_test, y_pred)
                        accuracies.append(accu)
                    accuracy_value = accuracies
                    return accuracy_value
        except Exception as e:
            messagebox.showerror("Error", "Please train first.")
    else:
        messagebox.showerror("Error", "Please train first.")
    return None

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def accu_win():
    accuracy = accu()
    if accuracy is not None:
        window_accu = tk.Toplevel(window)
        window_accu.title("Accuracy")

        window_width = 560 
        window_height = 400
        x = 260
        y = 200
        window_accu.geometry(f"{window_width}x{window_height}+{x}+{y}")
        if isinstance(accuracy, (int, float)):
            label_accu = tk.Label(window_accu, 
                                  text=f"Accuracy: {accuracy}",
                                  font=("AppleSystemUIFont", 20)
                                  )
            label_accu.pack()
        elif isinstance(accuracy, list):
            import matplotlib.pyplot as plt
            trees = list(range(1, len(accuracy) + 1))
            for widget in window_accu.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    widget.get_tk_widget().destroy()
            fig, ax = plt.subplots()
            ax.plot(trees, accuracy, marker='o')
            ax.set_xlabel('Index of Tree')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy vs  Trees')
            ax.grid(True)
            canvas = FigureCanvasTkAgg(plt.gcf(), master=window_accu)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

accu_but = tk.Button(master=model_buttons, text="Accuracy", command=accu_win)

train_but.grid(row=0,column=0, sticky="e", padx=0, ipadx=0)
plot_but.grid(row=0, column=1, padx=0, ipadx=0)
accu_but.grid(row=0, column=2, padx=0, ipadx=0)


window.mainloop()
