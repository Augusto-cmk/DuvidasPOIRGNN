from enum import Enum

class SequencesGenerationForPoiCategorizationSequentialBaselinesConfiguration(Enum):

    SEQUENCES_SIZE = ("sequences_size", 10)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]