from data import hugging_face_data


class data_prep:
    def __init__(self, language: str, text_type: str):
        self.language = language
        self.text_type = text_type
    def prepare_data(self):
        self.data_set = hugging_face_data.get_data(self.language,self.text_type)
        split = self.data_set.train_test_split(test_size=0.2, seed=42)
        self.train = split["train"]
        test = split["test"]
        return self.train,test

    def get_col_name(self, language:str):
        available_cols_train = self.train.column_names
        if "text" in available_cols_train:
            target_col = "text"
        elif language in available_cols_train:
            target_col = language
        else:
            target_col = available_cols_train[0]

        return target_col
