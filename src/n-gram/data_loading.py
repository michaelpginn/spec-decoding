from data import hugging_face_data


class data_prep:
    def __init__(self, language: str, text_type: str):
        self.language = language
        self.text_type = text_type
    def prepare_data(self):
        self.data = hugging_face_data.get_data(self.language,self.text_type)
        split = self.data.train_test_split(test_size=0.2, seed=42)
        train = split["train"]
        test = split["test"]
        return train,test
