# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random


st.title('Traffic Volume Predictor') 




# Display the image
st.write("Utilize our advanced ML app to predict traffic volume") 
st.image('traffic_image.gif', width = 600)


st.info("""
    Please choose a data input method to proceed
    """)
alpha = st.slider("**Select alpha value (confidence level)**", min_value=0.01, max_value=0.30, value=0.10, step=0.01,
                          help="e.g., 0.10 = 90% prediction interval")
alpha_percent = 1-alpha
# Load the pre-trained model from the pickle file
xgb_traffic = open('xgb_traffic.pickle', 'rb') 
clf = pickle.load(xgb_traffic) 
xgb_traffic.close()

# Create a sidebar for input collection
st.sidebar.image('traffic_sidebar.jpg', width = 200)
st.sidebar.header('Input FEatures')
st.sidebar.write('You can either upload your data file or input the features manually')



#---------------------------------------------------------------------------------------------
# Using Default (Original) Dataset to Automate Few Items
#---------------------------------------------------------------------------------------------

# Load the default dataset
df = pd.read_csv('Traffic_Volume.csv')
#df = df.head(200)
df['date_time'] = pd.to_datetime(df['date_time'])
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.dayofweek
df['hour'] = df['date_time'].dt.hour
df = df.drop(columns=['date_time'])
default_df = df

default_df = default_df.dropna().reset_index(drop = True) 
# NOTE: drop = True is used to avoid adding a new column for old index

novolume =  default_df.drop(columns = ['traffic_volume'])

with st.sidebar.expander("Option 1: 📂 CSV Upload"):
    traffic_file = st.file_uploader("Upload a traffic CSV", type=["csv"])
    st.write("Sample Data Format for Upload")
    st.dataframe(novolume.head(5), use_container_width=True)
    st.warning("Ensure your uploaded file has the same column names and fata types as shown above")

with st.sidebar.expander("Option 2: 📝 Manual Input (Form)"):
    with st.form("traffic_input_form"):  # <— create a form with a unique key
        st.write("Enter traffic features below:")

        holiday = st.selectbox("Holiday",default_df['holiday'].unique())
        temp = st.number_input(
            'Temperature',
            min_value=float(default_df['temp'].min()),
            max_value=float(default_df['temp'].max()),
            step=0.01,
        )
        rain = st.number_input(
            'Rain',
            min_value=float(default_df['rain_1h'].min()),
            max_value=float(default_df['rain_1h'].max()),
            step=.1,
        )
        snow = st.number_input(
            'Snow',
            min_value=float(default_df['snow_1h'].min()),
            max_value=float(default_df['snow_1h'].max()),
            step=.1,
        )
        clouds = st.number_input(
            'Clouds',
            min_value=float(default_df['clouds_all'].min()),
            max_value=float(default_df['clouds_all'].max()),
            step=.1,
        )
        weather = st.selectbox("Weather",default_df['weather_main'].unique())
        # date_time = st.datetime_input("Select a Date and Time", 
        #                               min_value = default_df['date_time'].min(),
        #                             max_value = default_df['date_time'].max() )

       # Get min and max datetime from dataset
        hour = st.selectbox("Hour", sorted(default_df['hour'].astype(int).unique()))
        month = st.selectbox("Month", sorted(default_df['month'].astype(int).unique()))
        weekday = st.selectbox("Weekday", sorted(default_df['weekday'].astype(int).unique()))


        # 🔘 Add the submit button here
        submitted = st.form_submit_button("✅ Submit Form Data")



# If no file is provided, then allow user to provide inputs using the form
if traffic_file is None:
        # Encode the inputs for model prediction
    
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['traffic_volume'])
    
    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, 
                                     clouds, weather, month,weekday,hour]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)


    # Convert datetime column into numeric features
    # if 'date_time' in user_encoded_df.columns:
    #     user_encoded_df['date_time'] = pd.to_datetime(user_encoded_df['date_time'])

    #     user_encoded_df['hour'] = user_encoded_df['date_time'].dt.hour
    #     user_encoded_df['dayofweek'] = user_encoded_df['date_time'].dt.dayofweek
    #     user_encoded_df['month'] = user_encoded_df['date_time'].dt.month

    # Drop the raw datetime column
    #user_encoded_df = user_encoded_df.drop(columns=['date_time'])

    # Using predict() with new data provided by the user
    #new_prediction = clf.predict(user_encoded_df)

    # # Show the predicted species on the app
    # st.subheader("Price")

    y_pred, y_pis = clf.predict(user_encoded_df, alpha=alpha)

    # # Display results
    lower, upper = y_pis[0]
    lower_val = float(np.ravel(lower)[0])
    upper_val = float(np.ravel(upper)[0])
    # st.subheader("Price Results")
    # st.write(f"**Expected Price:** {y_pred[0]:.2f}")
    # st.write(f"**90% Prediction Interval:** ({lower_val:.2f}, {upper_val:.2f})")
    # Green header text
    st.markdown("<h2 style='color: green;'>Predicting Traffic Volume...</h2>", unsafe_allow_html=True)

    # Predicted price (big bold font)
  
    st.markdown(f"<h3>Predicted Volume</h3>", unsafe_allow_html=True)
    pred_val = float(np.ravel(y_pred)[0])
    st.markdown(f"<h1 style='font-weight: bold;'>${pred_val:,.2f}</h1>", unsafe_allow_html=True)

    # Blue box for prediction interval
    st.markdown(
        f"""
        <div style="
            background-color:#e6f0ff;
            padding:10px;
            border-radius:5px;
            border:1px solid #99c2ff;
            width:fit-content;
        ">
            <b>Prediction Interval ({alpha_percent:,.2f}%):</b> [{lower_val:,.2f}, {upper_val:,.2f}]
        </div>
        """,
        unsafe_allow_html=True
    )

else:
   # Loading data
   st.success("""
    ### CSV file uploaded successfully
    """)

   user_df = pd.read_csv(traffic_file) # User provided data
   original_df = default_df # Original data to create ML model

   user_df = user_df.dropna().reset_index(drop = True) 
   user_df = user_df[novolume.columns]

   combined_df = pd.concat([novolume, user_df], axis = 0)
   encode_df = default_df.drop(columns=['traffic_volume']).copy()
   encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, clouds, weather, month, weekday, hour]
   encode_dummy_df = pd.get_dummies(encode_df)

    # Align to training columns
   with open("xgb_feature_cols.pickle", "rb") as f:
       feature_cols = pickle.load(f)

   user_encoded_df = user_encoded_df.reindex(columns=feature_cols, fill_value=0)

   y_pred, y_pis = clf.predict(user_encoded_df, alpha=alpha)

   lowers = y_pis[:, 0]
   uppers = y_pis[:, 1]

   # Convert to floats
   lowers = lowers.astype(float)
   uppers = uppers.astype(float)
   y_pred = y_pred.astype(float)

   # Add results to user DataFrame
   user_df["Predicted Volume"] = y_pred
   user_df["Lower Volume Limit"] = lowers
   user_df["Upper Volume Limit"] = uppers
   
   # Display nicely
   st.subheader("📈 Predicted Volume with Intervals")
   st.dataframe(user_df, use_container_width=True)

# Showing additional items in tabs
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])


# Tab 1: Feature Importance Visualization
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

# Tab 2: Visualizing Histogram of Residuals
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residuals_dist.svg')
    st.caption("Distribution of residuals to evaluate prediction quality")

# Tab 3: Predicted vs. Actual
with tab3:
    st.write("### Plot of Predicted vs. Actual")
    st.image('pred_vs_act.svg')
    st.caption("Visual comparison of predicted actual values.")

# Tab 4: Coverage plot
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_plot.svg')
    st.caption("Range of predictions with confience intervals.")
