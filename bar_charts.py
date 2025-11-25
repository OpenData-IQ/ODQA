import matplotlib.pyplot as plt

dcat_themes = {
     "Education, culture and sport": 21,
     "Population and society": 39,
     "Science and technology": 2,
     "Transport": 10,
     "Government and public sector": 24,
     "Agriculture, fisheries, forestry and food": 5,
     "Economy and finance": 21,
     "Environment": 24,
     "Energy": 4,
     "Regions and cities": 10,
     "Health": 14,
     "International issues": 2,
     "Justice, legal system and public safety": 2
}


question_types = {
    "simple": 21,
    "simple with restriction": 53,
    "multi hop": 11,
    "post processing heavy": 25,
    "set": 21,
    "false premise": 21,
    "aggregation": 27,
    "comparison": 21
}


def generate_plot(dictionary: dict):
    value_sum = sum(dictionary.values())
    pairs = [(key,(value/value_sum) * 100) for key, value in dictionary.items()]
    pairs.sort(key=lambda x: x[1], reverse=False)
    themes, values = zip(*pairs)
    plt.figure(figsize=(8,4))
    plt.barh(themes,values,color="orange")
    #plt.barh(themes, values)
    plt.xlabel("Share in %")
    plt.title("DCAT Themes")
    plt.tight_layout()
    plt.show()


generate_plot(dcat_themes)
#generate_plot(question_types)
