import pickle
from models.Breast_Cancer.BREAST_CANCER import Breast_Cancer  # Import your Pydantic model class

# Load the trained classifier model
classifier = pickle.load(open("models/Breast_Cancer/b_cancer.pkl", "rb"))

if __name__ == "__main__":
    # Create an instance of your Pydantic model with test data
    test_data = Breast_Cancer(
        clump_thickness=1.0,
        uniform_cell_size=1.0,
        uniform_cell_shape=1.0,
        marginal_adhesion=1.0,
        single_epithelial_size=1.0,
        bare_nuclei=1.0,
        bland_chromatin=1.0,
        normal_nucleoli=1.0,
        mitoses=1.0
    )

    # Make predictions using the classifier
    prediction = classifier.predict([[
        test_data.clump_thickness,
        test_data.uniform_cell_size,
        test_data.uniform_cell_shape,
        test_data.marginal_adhesion,
        test_data.single_epithelial_size,
        test_data.bare_nuclei,
        test_data.bland_chromatin,
        test_data.normal_nucleoli,
        test_data.mitoses
    ]])

    print("Prediction:", prediction)
