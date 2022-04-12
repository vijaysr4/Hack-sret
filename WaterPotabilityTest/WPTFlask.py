from contextlib import nullcontext
from flask import Flask , render_template, url_for, redirect, request

app = Flask(__name__) 

@app.route("/")
def home():
    return render_template('Home.html')

@app.route("/dataform", methods=["POST", "GET"])
def dataform():
    return render_template('Dataform.html')

@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == 'POST':
        # water_Ph = request.form.get("water_Ph")
        # water_Hardness = request.form.get("water_Hardness")
        # water_Solids = request.form.get("water_Solids")
        # water_Chloramine = request.form.get("water_Chloramine") 
        # water_Sulfate = request.form.get("water_Sulfate")
        # water_Conductivity = request.form.get("water_Conductivity")
        # water_Organic_Carbon = request.form.get("water_Organic_Carbon")
        # water_Trihalomethanes = request.form.get("water_Trihalomethanes")
        # water_Turbidity = request.form.get("water_Turbidity")
        
        # print("=================")
        # print(water_Ph)
        # print(water_Hardness)
        # print(water_Solids)
        # print(water_Chloramine)
        # print(water_Sulfate)
        # print(water_Conductivity)
        # print(water_Organic_Carbon)
        # print(water_Trihalomethanes)
        # print(water_Turbidity)
        # print("=================")
        
        Dict = {"Ph" : request.form.get("water_Ph"),
                "Hardness" : request.form.get("water_Hardness"),
                "Solids" : request.form.get("water_Solids"),
                "Chloramine" : request.form.get("water_Chloramine") ,
                "Sulfate" : request.form.get("water_Sulfate"),
                "Conductivity" : request.form.get("water_Conductivity"),
                "Organic_Carbon" : request.form.get("water_Organic_Carbon"),
                "Trihalomethanes" : request.form.get("water_Trihalomethanes"),
                "water_Turbidity" : request.form.get("water_Turbidity"),
                }
 
        print("=================")
        print(Dict["Ph"])
        print(Dict["Hardness"])
        print(Dict["Solids"])
        print(Dict["Chloramine"])
        print(Dict["Sulfate"])
        print(Dict["Conductivity"])
        print(Dict["Organic_Carbon"])
        print(Dict["Trihalomethanes"])
        print(Dict["water_Turbidity"])
        print("=================")
        
    water_pt = True
    return render_template('Result.html', pt = "Water Is Drinkable") if water_pt == True else render_template('result.html', pt = "Water is Not Drinkable")
 
if __name__ =='__main__':  
    app.run(debug = True)