# This program receives the tagger type and the path to a test file
# as command line parameters and outputs the POS tagged version of that file.
import os
import argparse
import nltk
from nltk.tag import hmm 
from nltk.probability import LidstoneProbDist
from nltk.tag.util import untag
from nltk.tag import brill, brill_trainer, UnigramTagger, BigramTagger, TrigramTagger, RegexpTagger

def hmm_tagger(train_data):
    trainer = hmm.HiddenMarkovModelTrainer()
    estimator = None
    if estimator is None:
        def estimator(fd, bins):
            return LidstoneProbDist(fd, 0.1, bins)
    tagger = trainer.train_supervised(labelled_sequences=train_data, estimator=estimator)

    return tagger

def hmm_test(test_data, train_data):

    untagged_test_data = []   
    # untagging the test data
    for x in test_data:
        l = untag(x)
        untagged_test_data.append(l)

    tagger = hmm_tagger(train_data)
    tagged_tokens = []

    for sent in untagged_test_data:
        tagged_token = tagger.tag(sent)
        tagged_tokens.append(tagged_token)
    
    accuracy = tagger.accuracy(test_data)
    print(f"The accuracy of this HMM tagger is: {accuracy}")
    return tagged_tokens

    
def brill_tagger(train_data, **kwargs):
    templates = [
            brill.Template(brill.Pos([-1])),
            brill.Template(brill.Pos([1])),
            brill.Template(brill.Word([-1])),
            brill.Template(brill.Word([1])),
            brill.Template(brill.Pos([-2])),
            brill.Template(brill.Pos([2])),
            brill.Template(brill.Word([-2])),
            brill.Template(brill.Word([2])),
            brill.Template(brill.Pos([-2, -1])),
            brill.Template(brill.Word([-2, -1])),
            brill.Template(brill.Pos([1, 2])),
            brill.Template(brill.Word([1, 2])),
            brill.Template(brill.Pos([-3, -2, -1])),
            brill.Template(brill.Word([-3, -2, -1])),
            brill.Template(brill.Pos([1, 2, 3])),
            brill.Template(brill.Word([1, 2, 3])),
            brill.Template(brill.Pos([-1]), brill.Pos([1])),
            brill.Template(brill.Word([-1]), brill.Word([1])),
            ]
      
    # Using BrillTaggerTrainer to train 
    backoff = RegexpTagger([
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    (r'.*able$', 'JJ'),                # adjectives
    (r'.*ness$', 'NN'),                # nouns formed from adjectives
    (r'.*ly$', 'RB'),                  # adverbs
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # past tense verbs
    (r'.*', 'NN')                      # nouns (default)
    ])
    
    initial_tagger =UnigramTagger(train_data, backoff = backoff)
    trainer = brill_trainer.BrillTaggerTrainer(
            initial_tagger, templates)
      
    brill_tagger = trainer.train(train_data, max_rules=200, min_acc=0.9)

    return brill_tagger


def brill_test(test_data, train_data):

    untagged_test_data = []
    for x in test_data:
        l = untag(x)
        untagged_test_data.append(l)

    tagger = brill_tagger(train_data)
    tagged_tokens = []

    for sent in untagged_test_data:
        tagged_token = tagger.tag(sent)
        tagged_tokens.append(tagged_token)
    
    accuracy = tagger.accuracy(test_data)
    
    print(f"The accuracy of this brill tagger is: {accuracy}")

    return tagged_tokens


# This function accepts input directory path as an argument
# Opens the input file and formats the read data to the required format, [[(word, tag)]]
def read_files(file_location):
    read_data = []
    with open(file_location,'r', encoding="utf-8") as x:
        lines = x.read().split("\n\n")  # List containing each line read from file location
        for line in lines:
            pairs = line.split("\n")
            sent = []
            for token in pairs:
                if token != "":
                    pair = token.split(" ")
                    pair = tuple(pair)
                    sent.append(pair)
            read_data.append(sent)

    return read_data


# This function accepts output directory path and the output data dict. as arguments,
# Opens the output file and writes the output data to the csv file.
def output_file(output_location, data_out):
    # To open the output csv file in write mode
    with open(output_location, 'w', encoding='UTF8', newline='') as f:
        for i in data_out:
           f.write('\n'.join(f'{j[0]} {j[1]}' for j in i))
           f.write('\n\n')
                
def evaluate_tagger(input, output):
    # the function helps us check what words were incorrectly tagged
    errors = {}
    output_data = read_files(output)
    input_data = read_files(input)
    for sentence in range(len(input_data)):
        for tupple in range(len(input_data[sentence])):
            if input_data[sentence][tupple][0] == output_data[sentence][tupple][0] and input_data[sentence][tupple][1] != output_data[sentence][tupple][1]:
                lst = []
                lst.append('correct = ' + input_data[sentence][tupple][1] + "    " + 'wrong = ' + output_data[sentence][tupple][1])
                errors[input_data[sentence][tupple][0]] = lst
                lst = []
    
    for error, value in errors.items():
        print(error, value)

def main():
    # To create an argument parser object
    parser = argparse.ArgumentParser()

    # To add arguments to the parser object
    parser.add_argument("--tagger", help="Please provide the location of the tagger")
    parser.add_argument("--train", help="Please provide the location of the train data file")
    parser.add_argument("--test", help="Please provide the location of the test file")
    parser.add_argument("--output", help="Please provide the location of the output")
    parser.add_argument("-v", "--verbosity", help="Increase output verbosity", action= "store_true" )

    # parses arguments through the parse_args() method. This will inspect the command line, 
    # convert each argument to the appropriate type and then invoke the appropriate action.
    args = parser.parse_args()
    
    tagger = args.tagger  # Variable to store tagger selection
    test = args.test  # Variable to store test file path 
    train = args.train  # Variable to store train file path 
    output = args.output  # Variable to store output file path

    if args.verbosity:  # Helps the user check what input and output file paths they have provided
        print(f"the location of the  tagger is {tagger} ")
        print(f"the location of the train data is {train} ")
        print(f"the location of the output file is {test} ")
        print(f"the type of model specified to train is {output} ")

    train_data = read_files(train)
    test_data = read_files(test)

    #if hmm is called then we call the hmm_test function and write the output to file
    if tagger == 'hmm':
        data_out=hmm_test(test_data,train_data)
        output_file(output,data_out)

    #if brill is called then we call the hmm_test function and write the output to file
    elif tagger == 'brill':
        data_out = brill_test(test_data, train_data)
        output_file(output,data_out)

    #evaluate_tagger(test, output)      #uncomment to see how we were checking for incorrect tags being predicted
main()

