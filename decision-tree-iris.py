from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

# #3 - cross validation
def max_depth_cross_validation(X_train,X_test,y_train,y_test,depths):
    test_accuracy = []
    train_accuracy = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        train_acc = accuracy_score(y_train,train_predictions)
        train_accuracy.append(train_acc)
        test_acc = accuracy_score(test_predictions,y_test)
        test_accuracy.append(test_acc)
        cm = confusion_matrix(y_test,test_predictions)
        # #2 - output the accuracy score of the model on the test data
        print(f"Testing data accuracy for the depth of {depth}: {test_acc}")
        # #4 - print ocnfusion matrix
        print(f"Testing data confusion matrix for depth = {depth}:")
        print(cm)
    
    return test_accuracy,train_accuracy

##plot the accuracy of model to visualize accuracy vs model_depth
def plot_accuracy(test_acc,train_acc,max_depth):
    plt.plot(max_depth, test_acc, label='Testing Error') # Plot testing error over domain
    plt.plot(max_depth, train_acc, label='Training Error') # Plot training error over domain
    plt.xlabel('Maximum Depth') 
    plt.ylabel('Accuracy') 
    plt.legend() # Show plot labels as legend
    plt.show() # Show graph

def decision_tree_iris():  
    data = load_iris()
    
    #do we want to add bias to the features?
    features = data["data"]
    labels = data["target"]

    max_depth = [1,5,10]
    
    # #1 - split the data into training and test data
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.20,random_state=40)
    
    #cross-validation for max_depth
    test_acc,train_acc = max_depth_cross_validation(X_train,X_test,y_train,y_test,max_depth)
    
    #fetch result of cross_validation. It will be the max_depth with the highest accuracy
    highest_index = test_acc.index(max(test_acc))
    best_depth = max_depth[highest_index]
    print(f"Depth with highest accuracy: {best_depth}")
    
    plot_accuracy(test_acc,train_acc,max_depth)


if __name__ =="__main__":
    decision_tree_iris()