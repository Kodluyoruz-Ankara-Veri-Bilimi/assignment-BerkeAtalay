
import pandas as pd
import numpy as np   
from IPython.display import display

class information:
    
    
    def __init__(self, path, file, data= pd.DataFrame(),catlist = [], catdata= pd.DataFrame()):
        self.path = path
        self.file_name = file
        self.data = data
        self.catlist = catlist
        print("Class object initialized") 
       
       
    def _load_data_info(self):
        #kendime not: file name için . dan ayır sondakini al eğer csv ise read csv değilse ne olduğuna göre güncelle
        typ = self.file_name.split(".")[-1]
        
        if typ == "csv":
            data = pd.read_csv(self.path + self.file_name)
            
        elif typ == "xlsx":
            data = pd.read_excel(self.path + self.file_name)
        
        else:
            print("This file type is not supported please use your own methods or contact me")
        
        print(self.file_name +" is loaded")
        return data
    
    def get_inf(self,load = True, method = ""):
        
        #prints basic information about raw data and collect column names with object type if method is entered does it current only have drop which drops rows with na valuex
        
        if load:
            data = self._load_data_info()
            self.data = data
            self.catdata = data
            
        if method == "drop":
            print(len(self.data))
            self.data = self.data.dropna()
            self.catdata = self.data
            print(len(self.data))
            
        display(self.data.head())
        print()
        display(self.data.info())
        print()
        display(self.data.describe().T)
        
        print("Columns names with object type are saved in .catlist if you want to turn them into dummies use cattodummy")
        self.catlist = list(self.data.select_dtypes(include=['object']).columns)
        
    
    def catinfo(self):
        display(self.data[self.catlist].head())
        
    def dropcolumn(self, column_list = []):
       
        for i in column_list:
            if i in self.catlist:
                self.catlist.remove(i)
            self.data = self.data.drop(i,axis = 1)
            self.catdata = self.catdata.drop(i,axis = 1)
            
 
    def cattodummy(self,column_names = []):
        
        #if user does not give spesific columns to transform it takes columns with object type
        if column_names == []:
            column_names = self.catlist
        
        print ("Looking for: " + str(column_names))
        
        
        
        if set(column_names).issubset(self.catdata): #checks if column_names in catdata to prevent errors
        
        
            for i in column_names: #transforms columns to dummies with adding orijinal name plus category name
                
                temp_df = pd.get_dummies(self.data[i])
                temp_columns = list(temp_df.columns)

                for j in range(len(temp_columns)):
                    temp_columns[j] = i+"_"+temp_columns[j]

                temp_df.columns = temp_columns
                print("Adding columns: ", list(temp_df.columns))
                self.catdata = pd.concat([self.catdata, temp_df], axis=1).drop(i,axis=1)

            print("Dummies entered as .catdata")
            display(self.catdata.head())
        else:
            #error message for columns
            print("Columns do not exist in catdata you may already transfer them to dummies, please check:")
            display(self.catdata.head())
            

    def choose_your_y (self,column_name):  #returns X , y depending on your selection of column name
         #!!!! Kendime NOT plotu optinel yap, sayısal durumlarda uzun sürüyor ve bir şeye benzemiyor
        #self.catdata[column_name].value_counts().plot.barh()
        y = self.catdata.drop(column_name, axis=1)
        display(y.head())
        return self.catdata[column_name], y
 

    
class models:
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        print("Model class 42")
        
    def change_x(self,x):
        self.x = x
        
    def model_selection(self,model_type = "", slvr = "", rs = 42):
        
        a="default"
        
        if model_type == "logres":
            from sklearn.linear_model import LogisticRegression
            
            if slvr:
                
                self.classifier = LogisticRegression(random_state = rs, solver = slvr)
                
                a = slvr
                
            
            else:
                self.classifier = LogisticRegression(random_state = rs)
            print("Logistic Regression Classifier is initiliazed with  " + a + " solver.")
        
        elif model_type == "pca":
            
            from sklearn.decomposition import PCA
            self.classifier = PCA()
            print("PCA is initilialized without specific number of components")
            
            
        else:
            print("Please Choose a classifier")
                
    
    def pca_proc(self,threshold=0.8, norm = True, split = True, rate = 0.2, rs = 42):
        
        
        #!!! Kendime not bu kısımları model_fit de aynı olanları model_preporc diye yeni bir foncvtionun içine al ve onu çağır
        
        if norm:
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            x = pd.DataFrame(sc_X.fit_transform(self.x),index = self.x.index)
        else:
            x = self.x
        
      
             
      
        X_pca = pd.DataFrame(self.classifier.fit_transform(x),index = x.index)
        
        if split:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_pca, self.y, test_size = rate, random_state = rs)
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        
        else:
            X_train = X_pca
            self.X_train = X_pca
            self.y_train = self.y
            self.X_test = X_pca
            self.y_train = self.y
            self.y_test  = self.y
            
        arr = np.cumsum(np.round(self.classifier.explained_variance_ratio_, decimals = 4)*100)
        num_var = sum((arr < threshold*100)) + 1 
        print('Pca sonrası değişken sayısı: ',num_var)
        X_pcad = self.X_train[:,0:num_var]
        self.X_train_pcad = X_pcad
        print("New X_train values are saved as X_train_pcad")
        
        
        

    def model_fit(self,norm = False, split = False, rate = 0.2, rs = 42):
        
        if norm:
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            x = sc_X.fit_transform(self.x)
        else:
            x = self.x
        
        if split:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(x, self.y, test_size = rate, random_state = rs)
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        
        else:
            X_train = x
            self.X_train = x
            self.y_train = self.y
            self.X_test = x
            y_train = self.y
            self.y_test  = self.y
        
        self.classifier.fit(X_train, y_train)
        
        print("Model fitted with normalization " + str(norm)+" and split " + str(split))
        

    def roc_curve(self):
        from sklearn.metrics import roc_auc_score,roc_curve
        import matplotlib.pyplot as plt

        logit_roc_auc = roc_auc_score(self.y_test, self.classifier.predict(self.X_test)) #alan
        fpr, tpr, thresholds = roc_curve(self.y_test, self.classifier.predict_proba(self.X_test)[:,1]) #roc curve

        plt.figure()
        plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Oranı')
        plt.ylabel('True Positive Oranı')
        plt.title('ROC')
        plt.show()
        
    def general_acc(self):
        y_test_pred = self.classifier.predict(self.X_test)
        from sklearn.metrics import confusion_matrix, accuracy_score
        print(accuracy_score(y_test_pred,self.y_test))
        print(confusion_matrix(self.y_test, y_test_pred))
        

    def predict_prob(self,pc = 0):
        
        from sklearn.metrics import classification_report, f1_score, confusion_matrix
        y_probs = self.classifier.predict_proba(self.X_test)
        y_probs = y_probs[:,1]
        if pc:
            y_pred = [1 if i > pc else 0 for i in y_probs]
            print(classification_report(self.y_test,y_pred))
            return f1_score(self.y_test,y_pred)
        else:      
            j = 0.1
            temp1 = []
            temp2 = []
            while j < 1:
                temp1.append(j)
                y_pred = [1 if i > j else 0 for i in y_probs]
                temp2.append(f1_score(self.y_test,y_pred))
                j += 0.05
               
            
            zipped_lists = zip(temp2, temp1)
            sorted_pairs = sorted(zipped_lists, reverse=True)
            imp = sorted_pairs[:3]
            
            for j in range(3):
                print("Using prob thrshold as " ,imp[j][1])
                y_pred = [1 if i > imp[j][1] else 0 for i in y_probs]
                print(classification_report(self.y_test,y_pred))
            
            print("For test data and with max f1 score: using prop thshold as: ",imp[0][1])
            y_pred = [1 if i > imp[0][1] else 0 for i in y_probs]
            print(confusion_matrix(self.y_test,y_pred))
            print("For train data and with max f1 score:using prop thshold as: ",imp[0][1])
            y_probs = self.classifier.predict_proba(self.X_train)[:,1]
            y_pred = [1 if i > imp[0][1] else 0 for i in y_probs]
            print(confusion_matrix(self.y_train,y_pred))
            
        
        
class data_classification():
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        print("Classification class")
        
    def split(self, rs=0):
        from sklearn.model_selection import train_test_split
        if rs:
            X_train, X_test, y_train , y_test = train_test_split(self.x,self.y,test_size = 0.3, random_state=rs)
        else:
            X_train, X_test, y_train , y_test = train_test_split(self.x,self.y,test_size = 0.3)
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def default_processes(self):
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report
        
        dtc = DecisionTreeClassifier()
        dtc.fit(self.X_train,self.y_train)
        y_pred = dtc.predict(self.X_test)
        print('Results for default decision tree')
        print(classification_report(self.y_test,y_pred))
        
        rfc = RandomForestClassifier()
        rfc.fit(self.X_train,self.y_train)
        y_pred = rfc.predict(self.X_test)
        print('Results for default random forest')
        print(classification_report(self.y_test,y_pred))
        
        mlpc = MLPClassifier()
        mlpc.fit(self.X_train,self.y_train)
        y_pred = mlpc.predict(self.X_test)
        print('Results for default MLP')
        print(classification_report(self.y_test,y_pred))
        
    def grid(self,method, params = {}):
        from sklearn.model_selection import  GridSearchCV
        if params:
            if method == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier()
                grid_params=params

            elif method == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                classifier = DecisionTreeClassifier()
                grid_params=params

            elif method == 'mlp':
                from sklearn.neural_network import MLPClassifier
                classifier = MLPClassifier()
                grid_params=params

            else:
                print('Unknown method')
                return
        else:
            if method == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier()
                grid_params = {"max_depth": [8,10,11,13,15,18],
                    "max_features": [5,10,15,20],
                     "n_estimators": [5,10,50,100,200,500],
                     "min_samples_split": [3,5,10],
                    "criterion":['entropy','gini']}

            elif method == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                classifier = DecisionTreeClassifier()
                grid_params = {"max_depth": range(1,10),
                    "min_samples_split": list(range(2,50)),
                    "criterion":['gini', 'entropy']}  

            elif method == 'mlp':
                from sklearn.neural_network import MLPClassifier
                classifier = MLPClassifier()
                grid_params = {
                    'alpha':[0.1,0.01,0.001,0.005,0.0001,0.00001],
                    'hidden_layer_sizes': [(10,10,10),(45,50,60),(25,35,45),(15,15)],
                    'solver': ['lbfgs','adam','sgd'],
                    'activation': ['relu','logistic','tanh','identity']  }
            else:
                print('Unknown method')
                return

        grid_cv = GridSearchCV(classifier, grid_params, cv=10, n_jobs=-1, verbose=2)
        grid_cv_model = grid_cv.fit(self.X_train,self.y_train)
        
        print("En iyi parametlerler: " + str(grid_cv_model.best_params_))
        
        
        
"""
class data_inf(data_load):
    # class that gives basic informations about data
    def __init__(self, df):
        data_load.__init__(self, df)

    def basicinf(self):
        print('First 5 data on this dataset: \n')
        print(self.data.head())
        print('Description of dataset: \n')
        print(self.data.describe())


class data_vis(data_inf):
    # class to visualize data
    def __init__(self, df):
        data_inf.__init__(self, df)

    def histo():  # histogram table
        pass

    def scatt():  # scatter plot
        pass

    def line():
        pass

    def box():
        pass

    def heat():
        pass


class data_preprec():
    def __init__(self):
        pass

    def dummie():
        pass

    def nn():
        pass

    def featurescale():
        pass


class helper():
    def __init__(self):
        pass

    def regression():  # and more functions to create model
        pass

    def fit(train):  # fit data using model
        pass

    def pred(test):  # predict
        pass

    def result():
        pass

"""
