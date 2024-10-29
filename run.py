import pickle

with open("dictionaries/bing_word_to_int.pkl", "rb") as file:
    res = pickle.load(file)

#sorted_by_values = sorted(res.items(), key=lambda item: item[1], reverse=True)
#print(sorted_by_values[:2000])
print("vocab", list(res.items())[-2000:])
