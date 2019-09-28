
def listener():
    input_model = input(
        "Please select the model to fit : svm or random_forest ? ")
    print("You selected the model " + input_model)

    cross_validation = False

    if input("Use cross validation ? (y/n) ") == 'y' :
        cross_validation = True
        print(" You selected cross validation ")

    print(" You did not select cross validation")
    print()
    return input_model, cross_validation
