<h1 align="center">Stock Predictor</h1>

This Stock Predictor was created to explore the power of deep learning in decoding stock market trends and to bridge my interests in technology and finance.

![Web App Screenshot](./Web%20App%20Picture.png)

## Dependencies
To run this project ensure you have these libraries installed:<br><br>
**NumPy:** Handles numerical operations and large arrays<br>
**Pandas:** Provides tools and data structures for analysis<br>
**yfinance:** Yahoo Finance's libary to use their information<br>
**Scikit-learn:** Essential for feature extraction and computing the similarity scores of the movies.<br>
**Keras** Libraries needed for building neural networks and training the deep learning model
**Streamlit:** Handles the front-end web interface<br>

To install all dependencies at once:

**pip install numpy pandas yfinance matplotlib pandas_datareader scikit-learn keras streamlit tensorflow**

## Usage
1. Copy the repository to your local machine.
2. Open the app.py file.
3. Install all dependencies
4. Enter "streamlit run app.py"
5. Wait for the web app to compile.
6. Enter a stock ticker name.
7. View the stock's historical data along with the predicted values on the displayed charts.
8. Experiment with different stock tickers to see varied predictions.

## Challenges Faced

When creating the deep learning model, the Keras libraries were throwing errors. After quite a bit of research, I realized that there is a problem between the new version of Python (3.11), the Apple M1 chip and TensorFlow(Keras). To fix this issue, I created a virtual environment kernel in Python 3.8 and used it to execute my model. Once this was done, there were no more errors related to this issue.
