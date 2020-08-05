
import pandas as pd
    
class information:
    
    
    def __init__(self, path, file, data= pd.DataFrame(),catlist = [], catdata= pd.DataFrame()):
        self.path = path
        self.file_name = file
        self.data = data
        self.catlist = catlist
        print("Class object initialized") 
       
       
    def _load_data_info(self):
        #kendime not: file name için . dan ayır sondakini al eğer csv ise read csv değilse ne olduğuna göre güncelle
        
        data = pd.read_csv(self.path + self.file_name)
        
        print(self.file_name +" is loaded")
        return data
    
    def get_inf(self,method = ""):
        
        #prints basic information about raw data and collect column names with object type if method is entered does it current only have drop which drops rows with na valuex
        
        data = self._load_data_info()
        
        if method == "drop":
            data = data.dropna()
        
        print(data.head())
        print()
        print(data.info())
        print()
        print(data.describe().T)
        
        self.data = data
        self.catdata = data
        self.catlist = list(data.select_dtypes(include=['object']).columns)    
        
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
            print(self.catdata.head())
        else:
            #error message for columns
            print("Columns do not exist in catdata you may already transfer them to dummies, please check:")
            print(self.catdata.head())
            

    def choose_your_y (self,column_name):  #returns X , y depending on your selection of column name
        self.catdata[column_name].value_counts().plot.barh()
        y = self.catdata.drop(column_name, axis=1)
        print(y.head())
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
            
        else:
            print("Please Choose a classifier")
            
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
            
            print("For test data and with max f1 score:")
            y_pred = [1 if i > imp[0][1] else 0 for i in y_probs]
            print(confusion_matrix(self.y_test,y_pred))
            print("For train data and with max f1 score:")
            y_probs = self.classifier.predict_proba(self.X_train)[:,1]
            y_pred = [1 if i > imp[0][1] else 0 for i in y_probs]
            print(confusion_matrix(self.y_train,y_pred))
            
        
        
        
        
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