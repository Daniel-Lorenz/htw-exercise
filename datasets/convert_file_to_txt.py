import pickle


def convert_file_to_txt(original, destination):
    with open(original, "rb") as infile:
        contents = pickle.load(infile, encoding="utf8")
    with open(destination, 'w', encoding="utf8") as output:
        for line in contents:
            output.write(line + "\n")


original_file_names = ('humorous_oneliners_win.pickle',
                       'proverbs_win.pickle',
                       'wiki_sentences_win.pickle',
                       'oneliners_incl_doubles_win.pickle',
                       'reuters_headlines_win.pickle')

new_file_names = ('humorous_oneliners.txt',
                  'proverbs.txt',
                  'wiki_sentences.txt',
                  'oneliners_incl_doubles.txt',
                  'reuters_headlines.txt')

for i in range(len(original_file_names)):
    convert_file_to_txt(original_file_names[i], new_file_names[i])
