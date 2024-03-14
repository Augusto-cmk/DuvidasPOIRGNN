from job.next_poi_category_prediction_sequences_generation_job import NextPoiCategoryPredictionSequencesGenerationJob

class PreprocessingData:
    def __init__(self):
        self.job = NextPoiCategoryPredictionSequencesGenerationJob()
    
    def do(self):
        self.job.start()
