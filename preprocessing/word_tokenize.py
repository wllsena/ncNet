from nltk import word_tokenize as nltk_word_tokenize


def word_tokenize(sent):
    if sent[:3] == '<n>':
        nl, cd = sent[3:].split('</n>', 1)
        nl_tokens = nltk_word_tokenize(nl)
        cd_tokens = cd.split()

        return ['<n>'] + nl_tokens + ['</n>'] + cd_tokens

    else:
        return sent.split()
