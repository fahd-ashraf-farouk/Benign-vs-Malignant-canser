import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as met
import matplotlib.pyplot as plt
import tkinter as tkk


#glbal variables to be used later
list1 = []
list2 = []
list3 = []
reads = None
title = None



#{"diagnosis":['M':0, 'B':1]}
deta = pd.read_csv("E:\\canser.csv")
deta = deta.drop(columns=["Unnamed: 32","id"])
deta = deta.replace({"diagnosis":{'M':0, 'B':1}})
print(deta.info())
x = deta.drop(columns="diagnosis")
y = deta["diagnosis"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=0)

def model():
    global title
    global reads
    model = RandomForestClassifier(n_estimators=5,random_state=0)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    acc= np.round(met.accuracy_score(y_test,pred),3)
    per = np.round(met.precision_score(y_test,pred),3)
    loo(pred)
    title = "RandomForestClassifier"
    reads.set(f"LogisticRegression acc: {acc} \n LogisticRegression per :{per}")

def model1():
    global title
    global reads
    model1 = DecisionTreeClassifier(random_state=0)
    model1.fit(x_train,y_train)
    pred1 = model1.predict(x_test)
    acc1 = np.round(met.accuracy_score(y_test,pred1),3)
    per1 = np.round(met.precision_score(y_test,pred1),3)
    title = "DecisionTreeClassifier"
    loo(pred1)
    reads.set(f"DecisionTreeClassifier acc: {acc1} \n DecisionTreeClassifier per :{per1}")

def model2():
    global title
    global reads
    model2 = LogisticRegression(fit_intercept=True,solver="liblinear",multi_class="auto",random_state = 0)
    model2.fit(x_train,y_train)
    pred2 = model2.predict(x_test)
    acc2 = np.round(met.accuracy_score(y_test,pred2),3)
    per2= np.round(met.precision_score(y_test,pred2),3)
    title = "LogisticRegression"
    loo(pred2)
    reads.set(f"LogisticRegression acc: {acc2} \n LogisticRegression per :{per2}")
def model3():
    global title
    global reads
    model3 = SVC(kernel="linear",probability=True,random_state = 0)
    model3.fit(x_train,y_train)
    pred3 = model3.predict(x_test)
    acc3 = np.round(met.accuracy_score(y_test,pred3),3)
    per3 = np.round(met.precision_score(y_test,pred3),3)
    title = "SVC"
    loo(pred3)
    reads.set(f"SVC acc: {acc3} \n SVC per :{per3}")

def model4():
    global title
    global reads
    model4 = KNeighborsClassifier(n_neighbors=4)
    model4.fit(x_train,y_train)
    pred4 = model4.predict(x_test)
    acc4 = np.round(met.accuracy_score(y_test,pred4),3)
    per4 = np.round(met.precision_score(y_test,pred4),3)
    title = "KNeighborsClassifier"
    loo(pred4)
    reads.set(f"KNeighborsClassifier acc: {acc4} \n KNeighborsClassifier per :{per4}")

def loo(x,y = y_test):
    global list1
    global list2
    global list3
    list1.clear(), list2.clear()
    v = []
    b = []
    for i in x :
        if i == 0:
            v.append(i)
        else:
            b.append(i)
    list1.append(len(v)),list1.append(len(b))
    v.clear(),b.clear()
    for i in y :
        if i == 0:
            v.append(i)
        else:
            b.append(i)
    list2.append(len(v)),list2.append(len(b))
    list3= [0, 1]

def line():
    global list1
    global list2
    global list3
    global title
    plt.plot(list3, list1, color="b", marker="o")
    plt.plot(list3, list2, color="k", marker="*")
    plt.grid()
    plt.xlabel("['M':0, 'B':1]")
    plt.ylabel("num of M or N")
    plt.title(title + "\n black true blue is predicted")
    plt.show()

def bar():
    global list1
    global list2
    global list3
    global title
    plt.bar(list3, list1, color="b", width=.12)
    plt.bar(list3, list2, color="k", width=.1)
    plt.grid()
    plt.xlabel("['M':0, 'B':1]")
    plt.ylabel("num of M or N")
    plt.title(title + "\n black true blue is predicted")
    plt.show()

def addTextLable(root):
    global reads
    reads = tkk.StringVar()
    reads.set("")
    my_reads = tkk.Label(root,textvariable=reads,borderwidth=20)
    my_reads.pack()


def exit():
    global root
    root.destroy()

#GUI
root = tkk.Tk()
q = tkk.Label(root,text= "ml project for predict B.canser or M.canser",borderwidth=20)
q.pack()
w = tkk.Button(root,text="RandomForestClassifier",command=model,borderwidth=15)
e = tkk.Button(root,text="DecisionTreeClassifier",command=model1,borderwidth=15)
r = tkk.Button(root,text="LogisticRegression",command=model2,borderwidth=15)
t = tkk.Button(root,text="SVC",command=model3,borderwidth=15)
u = tkk.Button(root,text="KNeighborsClassifier",command=model4,borderwidth=15)
i = tkk.Button(root,text="Exit",command=exit,borderwidth=15)
o = tkk.Button(root,text='Line plot',command=line , borderwidth=15)
p = tkk.Button(root,text='Bar plot',command=bar,borderwidth=15)

def pack_mod():
    w.pack(), e.pack(), r.pack(), t.pack(), u.pack()

def pack_exit():
    i.pack()

def print_pack():
    o.pack(), p.pack()


menu = tkk.Menu(root)
mod_men = tkk.Menu(menu)
mod_men.add_command(label="Models",command=pack_mod)
mod_pri = tkk.Menu(menu)
mod_pri.add_command(label="Plots",command=print_pack)
GUI_ex = tkk.Menu(menu)
GUI_ex.add_command(label="Exit",command=pack_exit)
menu.add_cascade(label="Models",menu=mod_men)
menu.add_cascade(label="Plots",menu=mod_pri)
menu.add_cascade(label="Exit",menu=GUI_ex)
addTextLable(root)
root.config(menu=menu)
root.mainloop()