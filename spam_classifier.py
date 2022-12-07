import pandas as pd



#change directory in "pd.read_csv" to location of spam.csv, for example 'C:\Users\USER_NAME\Desktop\spam.csv'

sms_spam = pd.read_csv(r'C:\Users\USERNAME\Desktop\spam.csv',encoding='ISO-8859-1')
sms_spam.rename(columns = {'v1':'Label', 'v2':'SMS'}, inplace = True)
sms_spam.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
sms_spam

# Training data for the spam classifier. For now, we can use short data, change to longer data later on.
train_spam = ['send us your password', 'review our website',
              'send your password', 'send us your account']
train_ham = ['Your activity report',
             'benefits physical activity', 'the importance vows']
test_emails = {'spam': ['renew your password', 'renew your vows'], 'ham': [
    'benefits of our account', 'the importance of physical activity']}


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
                spam_amount += 1
        #print(f'Number of spam emails with the word {word}: {spam_amount}')
        total_spam = len(train_spam)
        spamicity = (spam_amount+1)/(total_spam+2)
        #print(f'Spamicity of the word {word}: {spamicity} \n')
        spamicity_dict[word.lower()] = spamicity
    return spamicity_dict


"""What is smoothing? We do this so that we can avoid having a value of 0 for our 
spam training or ham training sets. This can cause a problem because having a 0 in
our numerator would lead to a 0 result. We can fix this by adding 1 to every word count,
so there will never wbe a zero word count. We then also have to add 2 to the denom to 
offset the change."""


def calculate_hamicity():
    ham_words = ham_vocab()
    hamicity_dict = {}
    for word in ham_words:
        ham_amount = 0
        for sentence in train_ham:
            if word in sentence:
                ham_amount += 1
        #print(f'Number of ham emails with the word \'{word}\': {ham_amount}')
        total_ham = len(train_ham)
        hamicity = (ham_amount+1) / (total_ham+2)  # apply smoothing
        #print(f'Hamicity of the word \'{word}\': {hamicity} \n')
        hamicity_dict[word.lower()] = hamicity
    return hamicity_dict


"""This computes the probability of any one email being spam, by dividing the total number 
of spam emails by the total number of all emails."""


def probability_of_spam():
    return len(train_spam) / (len(train_spam) + (len(train_ham)))
    # print(prob_spam)


"""This computes the probability of any one email being ham, by dividing the total number 
of ham emails by the total number of all emails."""


def probability_of_ham():
    return len(train_ham) / (len(train_spam) + (len(train_ham)))
    # print(prob_ham)


def get_distinct_words():
    tests = []
    for i in test_emails['spam']:
        tests.append(i)

    for i in test_emails['ham']:
        tests.append(i)

   # print(tests)

    distinct_words_as_sentences_test = []

    for sentence in tests:
        sentence_list = sentence.split()
        temp = []
        for word in sentence_list:
            temp.append(word)
        distinct_words_as_sentences_test.append(temp)

    # print(distinct_words_as_sentences_test)

    test_spam_tokenized = [
        distinct_words_as_sentences_test[0], distinct_words_as_sentences_test[1]]
    test_ham_tokenized = [
        distinct_words_as_sentences_test[2], distinct_words_as_sentences_test[3]]

    return test_spam_tokenized, test_ham_tokenized


def reduce_sentences():
    test_spam_tokenized, test_ham_tokenized = get_distinct_words()
    reduced_sentences_spam_test = []
    vocab_unique_spam = spam_vocab()
    vocab_unique_ham = ham_vocab()
    for sentence in test_spam_tokenized:
        word_list = []
        for word in sentence:
            if word in vocab_unique_spam:
                #print(f'\'{word}\', ok')
                word_list.append(word)
            elif word in vocab_unique_ham:
                #print(f'\'{word}\', ok')
                word_list.append(word)
            else:
                #print(f'\'{word}, word not present in labelled spam training data')
                pass
        reduced_sentences_spam_test.append(word_list)
    # print(reduced_sentences_spam_test)

    reduced_sentences_ham_test = []
    for sentence in test_ham_tokenized:
        word_list = []
        for word in sentence:
            if word in vocab_unique_ham:
                #print(f"'{word}', ok")
                word_list.append(word)
            elif word in vocab_unique_spam:
                #print(f"'{word}', ok")
                word_list.append(word)
            else:
                #print(f"'{word}', word not present in labelled ham training data")
                pass
        reduced_sentences_ham_test.append(word_list)

    # print(reduced_sentences_ham_test)

    return reduced_sentences_spam_test, reduced_sentences_ham_test


def stemming():
    reduced_sentences_spam_test, reduced_sentences_ham_test = reduce_sentences()
    test_spam_stemmed = []
    non_key = ['us',  'the', 'of', 'your']

    for email in reduced_sentences_spam_test:
        email_stemmed = []
        for word in email:
            if word in non_key:
                # print('remove')
                pass
            else:
                email_stemmed.append(word)
        test_spam_stemmed.append(email_stemmed)
    # print(test_spam_stemmed)

    test_ham_stemmed = []
    for email in reduced_sentences_ham_test:
        email_stemmed = []
        for word in email:
            if word in non_key:
                # print('remove')
                pass
            else:
                email_stemmed.append(word)
        test_ham_stemmed.append(email_stemmed)
    # print(test_ham_stemmed)

    return test_spam_stemmed, test_ham_stemmed


def mult(list_):        # function to multiply all word probs together
    total_prob = 1
    for i in list_:
        total_prob = total_prob * i
    return total_prob


def Bayes(email):
    dict_spamicity = calculate_spamicity()
    dict_hamicity = calculate_hamicity()
    total_spam = len(train_spam)
    total_ham = len(train_ham)
    probs = []
    for word in email:
        Pr_S = probability_of_spam()
        print('prob of spam in general ', Pr_S)
        try:
            pr_WS = dict_spamicity[word]
            print(f'prob "{word}"  is a spam word : {pr_WS}')
        except KeyError:
            # Apply smoothing for word not seen in spam training data, but seen in ham training
            pr_WS = 1/(total_spam+2)
            print(f"prob '{word}' is a spam word: {pr_WS}")

        Pr_H = probability_of_ham()
        print('prob of ham in general ', Pr_H)
        try:
            pr_WH = dict_hamicity[word]
            print(f'prob "{word}" is a ham word: ', pr_WH)
        except KeyError:
            # Apply smoothing for word not seen in ham training data, but seen in spam training
            pr_WH = (1/(total_ham+2))
            print(f"WH for {word} is {pr_WH}")
            print(f"prob '{word}' is a ham word: {pr_WH}")

        prob_word_is_spam_BAYES = (pr_WS*Pr_S)/((pr_WS*Pr_S)+(pr_WH*Pr_H))
        print('')
        print(
            f"Using Bayes, prob the the word '{word}' is spam: {prob_word_is_spam_BAYES}")
        print('###########################')
        probs.append(prob_word_is_spam_BAYES)
    print(f"All word probabilities for this sentence: {probs}")
    final_classification = mult(probs)
    if final_classification >= 0.5:
        print(
            f'email is SPAM: with spammy confidence of {final_classification*100}%')
    else:
        print(
            f'email is HAM: with spammy confidence of {final_classification*100}%')
    return final_classification


if __name__ == "__main__":
    # print("calc spamicity:")
    # calculate_spamicity()
    # print('\n')
    # print("calc hamicity:")
    # calculate_hamicity()
    # print("\n")
    # print("prob of spam:")
    # probability_of_spam()
    # print("prob of ham:")
    # probability_of_ham()
    # print("\n")
    # print("get distinct words: ")
    # get_distinct_words()
    # print("\n")
    # print("reduce sentences: ")
    # reduce_sentences()
    # print("\n")
    # print("stemming: ")
    # stemming()

    test_spam_stemmed, test_ham_stemmed = stemming()

    for email in test_spam_stemmed:
        print('')
        print(f"           Testing stemmed SPAM email {email} :")
        print('                 Test word by word: ')
        all_word_probs = Bayes(email)
        print(all_word_probs)

    for email in test_ham_stemmed:
        print('')
        print(f"           Testing stemmed HAM email {email} :")
        print('                 Test word by word: ')
        all_word_probs = Bayes(email)
        print(all_word_probs)









print(sms_spam.shape)
sms_spam.head()



sms_spam['Label'].value_counts(normalize=True)



# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)





training_set['Label'].value_counts(normalize=True)



test_set['Label'].value_counts(normalize=True)



# Before cleaning
training_set.head(3)



# After cleaning
training_set['SMS'] = training_set['SMS'].str.replace(
   '\W', ' ') # Removes punctuation
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head(3)



training_set['SMS'] = training_set['SMS'].str.split()

vocabulary = []
for sms in training_set['SMS']:
   for word in sms:
      vocabulary.append(word)

vocabulary = list(set(vocabulary))



len(vocabulary)




word_counts_per_sms = {'secret': [2,1,1],
                       'prize': [2,0,1],
                       'claim': [1,0,1],
                       'now': [1,0,1],
                       'coming': [0,1,0],
                       'to': [0,1,0],
                       'my': [0,1,0],
                       'party': [0,1,0],
                       'winner': [0,0,1]
                      }

word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()



word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
   for word in sms:
      word_counts_per_sms[word][index] += 1




word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()



training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()



# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1



# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

import re

def classify(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham: 
         p_ham_given_message *= parameters_ham[word]

   print('P(Spam|message):', p_spam_given_message)
   print('P(Ham|message):', p_ham_given_message)

   if p_ham_given_message > p_spam_given_message:
      print('Label: Ham')
   elif p_ham_given_message < p_spam_given_message:
      print('Label: Spam')
   else:
      print('Equal proabilities, have a human classify this!')


def classify_test_set(message):
   '''
   message: a string
   '''

   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham:
         p_ham_given_message *= parameters_ham[word]

   if p_ham_given_message > p_spam_given_message:
      return 'ham'
   elif p_spam_given_message > p_ham_given_message:
      return 'spam'
   else:
      return 'needs human classification'



test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()



correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
   row = row[1]
   if row['Label'] == row['predicted']:
      correct += 1

print("\n")
print('Correct:', correct, "\n")
print('Incorrect:', total - correct, "\n")
print('Accuracy:', correct/total, "\n")


