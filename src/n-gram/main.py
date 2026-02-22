from data_loading import data_prep
from datasets.packaged_modules import text

languages: list = [
    "Berber",
    "Cherokee",
    "Hawaiian",
    "Igbo",
    "Lakota",
    "Muskogee(Creek)",
    "Nepali",
    "Occitan",
    "Occitan",
    "Ojibwe",
    "Quechua",
    "Maya",
    "Tamazight",
    "Chinese",
]


def main():
    for language in languages:
        data = data_prep(language=language, text_type="mono")
        train,test = data.prepare_data()
    return 0


if __name__ == "__main__":
    main()
