import numpy as np

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration
from domain.next_poi_category_prediction_sequences_generation_domain import NextPoiCategoryPredictionSequencesGenerationDomain
from loader.file_loader import FileLoader
from configuration.next_poi_category_prediction_sequences_generation_configuration import SequencesGenerationForPoiCategorizationSequentialBaselinesConfiguration

from foundation.configuration.input import Input

class NextPoiCategoryPredictionSequencesGenerationJob:

    def __init__(self):
        self.file_loader = FileLoader()
        self.poi_categorization_configuration = NextPoiCategoryPredictionConfiguration()
        self.sequences_generation_for_poi_categorization_sequential_baselines_domain = NextPoiCategoryPredictionSequencesGenerationDomain(Input.get_instance().inputs['dataset_name'])

    def start(self):
        users_checkin_filename = Input.get_instance().inputs['users_steps_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        to_8_categories = Input.get_instance().inputs['to_8_categories']
        filename_8_categories = Input.get_instance().inputs['8_categories_filename']
        users_sequences_folder = Input.get_instance().inputs['users_sequences_folder']
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        userid_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['userid']
        category_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['category']
        locationid_column  = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['locationid']
        datetime_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['datetime']
        country_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['country']
        state_column = self.poi_categorization_configuration.DATASET_COLUMNS[1][dataset_name]['state']
        categories_to_int_osm = self.poi_categorization_configuration.CATEGORIES_TO_INT[1][dataset_name][categories_type]
        max_pois = self.poi_categorization_configuration.MAX_POIS[1]
        sequences_size = SequencesGenerationForPoiCategorizationSequentialBaselinesConfiguration.SEQUENCES_SIZE.get_value()

        users_checkin = self.sequences_generation_for_poi_categorization_sequential_baselines_domain.read_csv(users_checkin_filename, datetime_column)

        if dataset_name == "gowalla":
            users_checkin = users_checkin.query("state_name == 'Texas'")

        if to_8_categories == "yes":
            users_checkin = self.join_work_and_office_join_sport_leisure(users_checkin, category_column)
            print("cate: ", users_checkin['poi_resulting'].unique().tolist())
            self.file_loader.save_df_to_csv(users_checkin, filename_8_categories)

        users_sequences = self.sequences_generation_for_poi_categorization_sequential_baselines_domain.generate_sequences(users_checkin,
                                                                                                                          sequences_size,
                                                                                                                          max_pois,
                                                                                                                          userid_column,
                                                                                                                          category_column,
                                                                                                                          locationid_column,
                                                                                                                          datetime_column,
                                                                                                                          country_column,
                                                                                                                          state_column,
                                                                                                                          categories_to_int_osm,
                                                                                                                          dataset_name)

        self.sequences_generation_for_poi_categorization_sequential_baselines_domain.sequences_to_csv(users_sequences,users_sequences_folder)

    def join_work_and_office_join_sport_leisure(self, users_checkins, category_column):

        new_poi_resulting_list = []

        poi_resulting_list = users_checkins[category_column].tolist()

        for i in range(len(poi_resulting_list)):

            poi_resulting = poi_resulting_list[i]

            if poi_resulting == 'Office':
                poi_resulting = 'Work'
            elif poi_resulting == 'Sport':
                poi_resulting = 'Leisure'

            new_poi_resulting_list.append(poi_resulting)

        users_checkins['poi_resulting'] = np.array(new_poi_resulting_list)

        return users_checkins