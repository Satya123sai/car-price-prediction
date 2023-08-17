from flask import Flask,request,jsonify,render_template
import pandas as pd
app=Flask(__name__)
import pickle
car=pd.read_csv("carp.csv")
model=pickle.load(open("carpricemodel","rb"))

@app.route("/",methods=["GET"])
def home():
    
    company=car["company"].unique()
    transmission=car["transmission"].unique()
    owner=car["owner"].unique()
    seller=car["seller_type"].unique()
    fuel=car["fuel"].unique()
    return render_template("index.html",company=company,trans=transmission,owner=owner,seller=seller,fuel=fuel)

@app.route("/",methods=["POST"])
def predict():
    c=request.form["company"]
    t=request.form["transmission"]
    f=request.form["fuel"]
    s=request.form["seller_type"]
    o=request.form["owner"]
    y=request.form["year"]
    k=request.form["km_driven"]
    a=model.predict(pd.DataFrame([[c,t,f,s,o,y,k]],columns=['company', 'transmission', 'fuel', 'seller_type', 'owner', 'year',
       'km_driven']))[0]
    return render_template("index.html",pred=a)
if __name__=="__main__":
    app.run(port=3000,debug=True)
