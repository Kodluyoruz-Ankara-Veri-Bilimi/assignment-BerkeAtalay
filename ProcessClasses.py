# Kendime not: Inheritance gözden geçirelecek gereksiz olabilir veya tam tersi şekilde yapılması gerekebilir

class data_load:
    # class to load data (may combine with data_inf)
    def __init__(self, df):
        self.data = df


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


class data_preprec(df):
    def __init__(self):
        self.data = df

    def dummie():
        pass

    def nn():
        
        self.data = self.data.dropna()
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



