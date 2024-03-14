class CFGModel:
    def __init__(self,srcDir:str) -> None:
        """
        srcDir = Diretório onde se encontra os dados de entrada para o modelo POI-RGNN e o local de saída dos dados
        OBS: Nesse diretório, deve conter a seguinte configuração:
            - Uma pasta contendo os inputs chamada de "input"
            - Uma pasta chamada "output" para a saída dos dados do modelo
            - Um arquivo contendo os users_sequences.csv dentro de input
        """
        self.dir_input = f"{srcDir}/input"
    
    def get(self):
        return {
            "dataset_name":"gowalla",
            "categories_type":"7_categories",
            "users_sequences":f"{self.dir_input}/users_sequences.csv",
            "baseline":"poi_rgnne"
        }


class CFGSequence:
    def __init__(self,srcDir:str) -> None:
        """
        srcDir = Diretório onde se encontra os dados de entrada para o modelo POI-RGNN e o local de saída dos dados
        OBS: Nesse diretório, deve conter a seguinte configuração:
            - Uma pasta contendo os inputs chamada de "input"
            - Uma pasta chamada "output" para a saída dos dados do modelo
            - Um arquivo contendo os users_steps.csv dentro de input
        """
        self.dir_input = f"{srcDir}/input"
    
    def get(self):
        return {
            "users_steps_filename": f"{self.dir_input}/users_steps.csv",
            "base_dir":self.dir_input,
            "users_sequences_folder":f"{self.dir_input}/users_sequences.csv",
            "8_categories_filename":f"{self.dir_input}/users_steps.csv",
            "categories_type":"7_categories",
            "to_8_categories":"no",
            "dataset_name":"gowalla"
        }