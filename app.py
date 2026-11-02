import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, f1_score

st.set_page_config(
    page_title="SmartBike AI",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 SmartBike AI")
st.write("Predict hourly bike rentals or classify bike rental demand.")

@st.cache_resource
def train_models():

    df = pd.read_csv("hour.csv")

    features = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
        'workingday', 'weathersit', 'temp', 'atemp',
        'hum', 'windspeed'
    ]

    X = df[features]
    y = df['cnt']

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Multiple Linear Regression
    linear = LinearRegression()
    linear.fit(Xtr, ytr)

    y_pred_linear = linear.predict(Xte)

    rmse_linear = np.sqrt(
        mean_squared_error(yte, y_pred_linear)
    )

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)

    Xtrp = poly.fit_transform(Xtr)
    Xtep = poly.transform(Xte)

    polynomial = LinearRegression()
    polynomial.fit(Xtrp, ytr)

    y_pred_poly = polynomial.predict(Xtep)

    rmse_poly = np.sqrt(
        mean_squared_error(yte, y_pred_poly)
    )

    # Decision Tree Regression
    tree_r = DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    )

    tree_r.fit(Xtr, ytr)

    y_pred_tree = tree_r.predict(Xte)

    rmse_tree = np.sqrt(
        mean_squared_error(yte, y_pred_tree)
    )

    regression_errors = [
        rmse_linear,
        rmse_poly,
        rmse_tree
    ]

    best_r = int(np.argmin(regression_errors))

    regression_models = [
        linear,
        polynomial,
        tree_r
    ]


    # Define high-demand threshold
    threshold = ytr.quantile(0.75)

    yt = (ytr > threshold).astype(int)
    yv = (yte > threshold).astype(int)

    # Scale features for KNN
    scaler = StandardScaler()

    Xtrs = scaler.fit_transform(Xtr)
    Xtes = scaler.transform(Xte)

    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtrs, yt)

    # Decision Tree Classifier
    tree_c = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )

    tree_c.fit(Xtr, yt)

    # Gaussian Naive Bayes
    nb = GaussianNB()
    nb.fit(Xtr, yt)

    classification_models = [
        knn,
        tree_c,
        nb
    ]

    # Compare classification models using F1-score
    scores = [
        f1_score(yv, knn.predict(Xtes), zero_division=0),
        f1_score(yv, tree_c.predict(Xte), zero_division=0),
        f1_score(yv, nb.predict(Xte), zero_division=0)
    ]

    best_c = int(np.argmax(scores))

    return (
        regression_models,
        poly,
        best_r,
        classification_models,
        scaler,
        best_c,
        threshold,
        features
    )



try:
    (
        reg,
        poly,
        best_r,
        clf,
        scaler,
        best_c,
        threshold,
        features
    ) = train_models()

except FileNotFoundError:
    st.error("Please place hour.csv inside the data folder.")
    st.stop()


task = st.radio(
    "Select Prediction:",
    ["Regression", "High-Demand Classification"],
    horizontal=True
)


c1, c2, c3 = st.columns(3)

with c1:
    st.header("Time & Season")

    season = st.selectbox("Season", [1, 2, 3, 4])

    year = st.selectbox("Year", ["2011", "2012"])

    month = st.slider("Month", 1, 12, 6)

    hour = st.slider("Hour", 0, 23, 17)


with c2:
    st.header("Day Details")

    holiday = st.selectbox("Holiday?", [0, 1])

    weekday = st.selectbox(
        "Weekday",
        range(7),
        format_func=lambda x: [
            "Sunday", "Monday", "Tuesday",
            "Wednesday", "Thursday", "Friday", "Saturday"
        ][x]
    )

    workingday = st.selectbox("Working Day?", [1, 0])


with c3:
    st.header("Weather")

    weather = st.selectbox("Weather", [1, 2, 3, 4])

    temp = st.slider(
        "Temperature (°C)", 0.0, 41.0, 25.0
    )

    atemp = st.slider(
        "Feels Like (°C)", 0.0, 50.0, 26.0
    )

    hum = st.slider(
        "Humidity (%)", 0, 100, 50
    )

    wind = st.slider(
        "Wind Speed (km/h)", 0.0, 67.0, 15.0
    )


X = pd.DataFrame([[
    season,
    int(year) - 2011,
    month,
    hour,
    holiday,
    weekday,
    workingday,
    weather,
    temp / 41,
    atemp / 50,
    hum / 100,
    wind / 67
]], columns=features)



if st.button("🔮 Predict", type="primary"):

    if task == "Regression":

        if best_r == 0:

            result = reg[0].predict(X)[0]

            name = "Multiple Linear Regression"

        elif best_r == 1:

            X_poly = poly.transform(X)

            result = reg[1].predict(X_poly)[0]

            name = "Polynomial Regression"

        else:

            result = reg[2].predict(X)[0]

            name = "Decision Tree Regressor"

        st.success(
            f"Predicted Bike Rentals: {max(0, round(result))} bikes/hour"
        )

    else:

        if best_c == 0:

            X_scaled = scaler.transform(X)

            result = clf[0].predict(X_scaled)[0]

        else:

            result = clf[best_c].predict(X)[0]

        model_names = [
            "K-Nearest Neighbors",
            "Decision Tree Classifier",
            "Gaussian Naive Bayes"
        ]

        name = model_names[best_c]

        if result == 1:
            st.success("Predicted Demand: HIGH")
        else:
            st.info("Predicted Demand: NORMAL")

        st.caption(
            f"High-demand threshold: {threshold:.0f} rentals/hour"
        )

    st.caption(f"Selected model: {name}")
