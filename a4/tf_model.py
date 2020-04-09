# Converts the unicode file to ascii
def unicode_to_ascii(s):
    
    if unicodedata.category(c) != 'Mn':
        sentence = ''.join(c for c in unicodedata.normalize('NFD', s)
        return sentence

if __name__ == "__main__":
    a = unicode_to_ascii("aaa")