# Training data for the spam classifier. For now, we can use short data, change to longer data later on.
train_spam = ['send us your password', 'review our website', 'send your password', 'send us your account']  
train_ham = ['Your activity report','benefits physical activity', 'the importance vows']  
test_emails = {'spam':['renew your password', 'renew your vows'], 'ham':['benefits of our account', 'the importance of physical activity']}
def spam_vocab():
    # make a vocabulary of unique words that occur in known spam emails  
    vocab_words_spam = []  
    for sentence in train_spam:  
        sentence_as_list = sentence.split()  
        for word in sentence_as_list:  
            vocab_words_spam.append(word)     

    # Convert the list into a dictionary to get rid of duplicates.
    spam_words_unique = list(dict.fromkeys(vocab_words_spam))
    return spam_words_unique

def ham_vocab():
    # make a vocabulary of unique words that occur in known spam emails  
    vocab_words_ham = []  
    for sentence in train_ham:  
        sentence_as_list = sentence.split()  
        for word in sentence_as_list:  
            vocab_words_ham.append(word)     

    # Convert the list into a dictionary to get rid of duplicates.
    ham_words_unique = list(dict.fromkeys(vocab_words_ham))
    return ham_words_unique

"""How do we determine how spammy a word is? One way to do this is take the total
number of emails that have been labeled as spam, and count determine the frequency
of each word. This is essentially Bayes Rule. We can count how many spam emails have
the word "send" in them and divide that by the total number ospam emails. This gives 
us a measure of the words spamicity, or how likely the email is a spam email."""

def calculate_spamicity():
    spam_words = spam_vocab()
    spamicity_dict = {}
    for word in spam_words:
        spam_amount = 0
        for sentence in train_spam:
            if word in sentence:
                spam_amount+=1
        print(f'Number of spam emails with the word {word}: {spam_amount}')
        total_spam = len(train_spam)
        spamicity = (spam_amount+1)/(total_spam+2)
        print(f'Spamicity of the word {word}: {spamicity} \n')
        spamicity_dict[word.lower()] = spamicity

"""What is smoothing? We do this so that we can avoid having a value of 0 for our 
spam training or ham training sets. This can cause a problem because having a 0 in
our numerator would lead to a 0 result. We can fix this by adding 1 to every word count,
so there will never wbe a zero word count. We then also have to add 2 to the denom to 
offset the change."""

if __name__ == "__main__":
    calculate_spamicity()