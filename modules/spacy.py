import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import string
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from xgboost import XGBClassifier


class TrainingTest:
    def __init__(self) -> None:
        self.nlp = spacy.load('pt_core_news_sm')
        self.df = None

    def process(self, file):
        self.load_excel(file)
        self.df = self.pre_process()
        
        tfidf = self.tfidf()
        
        Y = self.df.Nível.values
        
        tfidf_train, tfidf_test, class_train, class_test = train_test_split(tfidf, Y, test_size=0.25)
        
        self.run_exps_train_test(tfidf_train, class_train, tfidf_test, class_test)
        final = self.run_exps_crossvalidation(tfidf, Y)
        grouped = final[['test_accuracy','test_f1_weighted', 'test_kappa']].groupby(final['model'])
        return grouped

    def load_excel(self, file):
        self.df = pd.read_excel(file)

    def pre_process(self):
        dataframe = self.remove_pontuacao(self.df, 'text')
        dataframe = self.remove_stopwords(self.df)
        dataframe = self.lematization(dataframe)
        
        return dataframe

    def remove_pontuacao(self, dataframe, colum_name):
        dataframe['Text_no_ponctuation_number'] = dataframe[colum_name].apply(lambda x: [token for token in x if token not in string.punctuation and not token.isnumeric()])
        dataframe['Text_no_ponctuation_number'] = dataframe['Text_no_ponctuation_number'].apply(lambda x: ''.join(x))
        
        return dataframe

    def remove_stopwords(self, dataframe):
        dataframe['Text_no_stopword'] = dataframe['Text_no_ponctuation_number'].apply(lambda x: [token.text.lower() for token in self.nlp(x) if (token.is_stop == False and len(token.text)>3)])
        dataframe['Text_no_stopword'] = dataframe['Text_no_stopword'].apply(lambda x: ' '.join(x))
        
        return dataframe

    def lematization(self, dataframe):
        dataframe['Text_lemma_no_stopword'] = dataframe['Text_no_stopword'].apply(lambda x: [token.lemma_ for token in self.nlp(x)])
        dataframe['Text_lemma_no_stopword'] = dataframe['Text_lemma_no_stopword'].apply(lambda x: ' '.join(x))
        dataframe['Text_lemma'] = dataframe['Text_no_ponctuation_number'].apply(lambda x: [token.lemma_ for token in self.nlp(x)])
        dataframe['Text_lemma'] = dataframe['Text_lemma'].apply(lambda x: ' '.join(x))
        
        return dataframe

    def tfidf(self):
        # Código para pegar os valores de uma coluna do dataframe (dataframe,nomedacoluna,.values)
        X = self.df.Text_lemma_no_stopword.values

        #Extração das features
        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_model = vectorizer.fit(X)

        return tfidf_model.transform(X)

    def run_exps_train_test(
        self,
        x_train: pd.DataFrame ,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame) -> pd.DataFrame:
        """
        Lightweight script to test many models and find winners
        :param x_train: train split
        :param y_train: training target vector
        :param x_test: test split
        :param y_test: test target vector
        :return: DataFrame of predictions
        """

        dfs = []
        #Modelos que serão avaliados (podem incluir quantos modelos quiserem)
        models = [
            ('LogReg', LogisticRegression()),
            ('RF', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier()),
            ('SVM', SVC(kernel="linear")),
            ('MNB', MultinomialNB()),
            ('Adaboost', AdaBoostClassifier()),
            ('XGB', XGBClassifier())
            ]

        results = []
        names = []
        #Métricas que serão avaliadas (podem incluir quantos métricas quiserem)
        kappa_scorer = make_scorer(cohen_kappa_score)
        scoring = {
                    'accuracy': 'accuracy',
                    'precision_weighted': 'precision_weighted',
                    'recall_weighted': 'recall_weighted',
                    'f1_weighted': 'f1_weighted',
                    'kappa' : kappa_scorer
                    }
                    
        #Nomes das classes, esse atributo é opcional, caso não seja incluido o modelo 
        #vai apresentar os valores de 0-n onde n é o número de classes.
        # target_names = ['ham', 'spam'] 

        for name, model in models:
            #em alguns casos é interessante se criar um classificador para cada classe
            #caso seja o caso descomentar linha abaixo
            #model = OneVsRestClassifier(model)
            clf = model.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            
    def run_exps_crossvalidation(
        self,
        x: pd.DataFrame ,
             y: pd.DataFrame) -> pd.DataFrame:
        """
        Lightweight script to test many models and find winners
        :param x: values vector
        :param y: target vector
        :return: DataFrame of predictions
        """

        dfs = []
        print("CARREGANDO MODELO")
        models = [
            ('LogReg', LogisticRegression()),
            ('RF', RandomForestClassifier()),
            ('KNN', KNeighborsClassifier()),
            ('MNB', MultinomialNB()),
            ('Adaboost', AdaBoostClassifier()),
            ('XGB', XGBClassifier())
            ]

        results = []
        names = []
        kappa_scorer = make_scorer(cohen_kappa_score)
        scoring = { 
                    'accuracy': 'accuracy',
                    'precision_weighted': 'precision_weighted',
                    'recall_weighted': 'recall_weighted',
                    'f1_weighted': 'f1_weighted',
                    'kappa' : kappa_scorer
                    }
        print("RODANDO")
        for name, model in models:
            print(name)
            kfold = model_selection.KFold(n_splits=10, shuffle=True)
            cv_results = model_selection.cross_validate(model, x, y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)

        final = pd.concat(dfs, ignore_index=True)
        return final
