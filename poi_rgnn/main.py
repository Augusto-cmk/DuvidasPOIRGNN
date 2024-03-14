from job.next_poi_category_prediction_job import NextPoiCategoryPredictionJob
from config.configLoader import CFGSequence,CFGModel
from foundation.configuration.input import Input
from preprocessing import PreprocessingData
from threading import Thread
import os
import time
import warnings
warnings.filterwarnings("ignore")

class POIRGNN:
    def __init__(self,path_data="Data"):
        self.__path_data = path_data
        self.loading = Thread(target=self.__carregamento)

    def preprocessamento(self):
        os.system("clear")
        print("Iniciando pré-processamento dos dados...")
        Input().set_inputs(CFGSequence(self.__path_data).get())
        PreprocessingData().do()
    
    def GNN(self):
        os.system("clear")
        print("Iniciando POIRGNN...")
        Input().set_inputs(CFGModel(self.__path_data).get())
        NextPoiCategoryPredictionJob().start()

    def __carregamento(self):
        os.system("clear")
        print("Aguarde uns minutos enquanto é computado as informações")
        print("\n")
        i = 0
        while True:
            os.system("tput rc")
            print("\n")
            if i == 0:
                string = "* "
            elif i == 1:
                string = "* * "
            elif i == 2:
                string = "* * * "
            else:
                string = "          "
            print(string)
            os.system("tput ed")
            i = (i + 1) % 4
            time.sleep(1)


# POIRGNN().preprocessamento()
POIRGNN().GNN()
