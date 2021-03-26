from flask import Flask, request, render_template #Web app framework
import numpy as np
import pandas as pd
#Model related libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#Plot related libraries
import matplotlib.pyplot as plt
from io import BytesIO
import base64


#Create app using flask
app = Flask(__name__)

@app.route('/')
def index():
    poly_order = request.args.get("poly_order", 1, type=int)
    
    #Load data
    airquality = pd.read_csv('airquality.csv')
    airquality.dropna(inplace=True)
    data = airquality[['Temp', 'Ozone']]
    plot_url, r2 = build_model(poly_order, data)
    return render_template('layout.html', tables=[data.head(6).to_html(classes='data', header=True)], r2=r2, plot_url=plot_url.decode('utf8'), chosen_order=poly_order)

def build_model(poly_order, data):

    x = data['Ozone']
    x = x[:, np.newaxis]
    y = data['Temp']
    y = y[:, np.newaxis]
    poly_features = PolynomialFeatures(degree = poly_order)
    x_poly = poly_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    #Training Error
    r2 = r2_score(y, y_poly_pred)

    #Sort axis and create plot
    plt.scatter(data['Ozone'], data['Temp'], s=10)
    sorted_zip = sorted(zip(data['Ozone'], y_poly_pred))
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.xlabel('Ozone (ppb)')
    plt.ylabel('Temp (Fahrenheit)')
    #Save plot to return to html
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue())
    return(plot_url, r2)

if __name__ == "__main__":	
		app.run(debug=True)