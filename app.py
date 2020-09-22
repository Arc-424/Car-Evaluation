# Core Pkg
import streamlit as st
import os

# EDA Pkgs
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg') 
import joblib

@st.cache
def load_data(dataset):
	df = pd.read_csv(dataset)
	return df


def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

buying_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
maint_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
doors_label = {'2': 0, '3': 1, '5more': 2, '4': 3}
persons_label = {'2': 0, '4': 1, 'more': 2}
lug_boot_label = {'small': 0, 'big': 1, 'med': 2}
safety_label = {'high': 0, 'med': 1, 'low': 2}
class_label = {'good': 0, 'acceptable': 1, 'very good': 2, 'unacceptable': 3}

# Get the Keys
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key


def main():
	"""Car Evaluation with ML Streamlit App"""

	st.title("Car Evaluation")
	st.subheader("Streamlit ML App")
	# st.image(load_image("cars_images/car1.jpg"),width=300, caption='Images')

	activities = ['EDA','Prediction','Gallery','About']
	choices = st.sidebar.selectbox("Select Activity",activities)

	if choices == 'EDA':
		st.subheader("EDA")
		data = load_data('car.data')
		st.dataframe(data.head(5))

		if st.checkbox("Show Summary of Dataset"):
			st.write(data.describe())

		# Show Plots
		if st.checkbox("Simple Value Plots "):
			st.write(sns.countplot(data['class']))
			# Use Matplotlib to render seaborn
			st.pyplot()

		# Show Columns By Selection
		if st.checkbox("Select Columns To Show"):
			all_columns = data.columns.tolist()
			selected_columns = st.multiselect('Select',all_columns)
			new_df = data[selected_columns]
			st.dataframe(new_df)

		if st.checkbox("Pie Plot"):
				all_columns_names = data.columns.tolist()
				if st.button("Generate Pie Plot"):
					st.write(data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
					st.pyplot()


	if choices == 'Prediction':
		st.subheader("Prediction")

		buying = st.selectbox('Select Buying Level',tuple(buying_label.keys()))
		maint = st.selectbox('Select Maintenance Level',tuple(maint_label.keys()))
		doors = st.selectbox('Select Doors',tuple(doors_label.keys()))
		persons = st.number_input('Select Num of Persons',2,10)
		lug_boot = st.selectbox("Select Lug Boot",tuple(lug_boot_label.keys()))
		safety = st.selectbox('Select Safety',tuple(safety_label.keys()))

		k_buying = get_value(buying,buying_label)
		k_maint = get_value(maint,maint_label)
		k_doors = get_value(doors,doors_label)
		# k_persons = get_value(persons,persons_label)
		k_lug_boot = get_value(lug_boot,lug_boot_label)
		k_safety = get_value(safety,safety_label)

		
		pretty_data = {
		"buying":buying,
		"maint":maint,
		"doors":doors,
		"persons":persons,
		"lug_boot":lug_boot,
		"safety":safety,
		}
		st.subheader("Options Selected")
		st.json(pretty_data)

		st.subheader("Data Encoded As")
		# Data To Be Used
		sample_data = [k_buying,k_maint,k_doors,persons,k_lug_boot,k_safety]
		st.write(sample_data)

		prep_data = np.array(sample_data).reshape(1, -1)

		model_choice = st.selectbox("Model Type",['logit','random forest','MLP classifier'])
		if st.button('Evaluate'):
			if model_choice == 'logit':
				predictor = load_prediction_models("logit_car_model.pkl")
				prediction = predictor.predict(prep_data)
				st.write(prediction)

			if model_choice == 'random forest':
				predictor = load_prediction_models("random_forest_car_model.pkl")
				prediction = predictor.predict(prep_data)
				st.write(prediction)

			if model_choice == 'MLP classifier':
				predictor = load_prediction_models("mlp_car_model.pkl")
				prediction = predictor.predict(prep_data)
				st.write(prediction)


			final_result = get_key(prediction,class_label)
			st.success(final_result)


if __name__ == '__main__':
	main()

