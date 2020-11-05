def convert_line_endings_to_windows(original, destination):
    outsize = 0
    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))
    print("Done. Saved %s bytes." % (len(content) - outsize))


original_file_names = ('humorous_oneliners.pickle',
                       'proverbs.pickle',
                       'wiki_sentences.pickle',
                       'oneliners_incl_doubles.pickle',
                       'reuters_headlines.pickle')

new_file_names = ('humorous_oneliners_win.pickle',
                  'proverbs_win.pickle',
                  'wiki_sentences_win.pickle',
                  'oneliners_incl_doubles_win.pickle',
                  'reuters_headlines_win.pickle')

for i in range(len(original_file_names)):
    convert_line_endings_to_windows(original_file_names[i], new_file_names[i])
