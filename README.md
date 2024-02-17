# Intro to NLP - Assignment 4

## Team
|Student name| CCID    |
|------------|---------|
|student 1   | varsani |
|student 2   |   gami  |

Please note that CCID is **different** from your student number.

## TODOs

In this file you **must**:
- [ x] Fill out the team table above. 
- [ x] Make sure you submitted the URL on eClass.
- [ x] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment.
- [x ] Provide clear installation and execution instructions that TAs must follow to execute your code.
- [x ] List where and why you used 3rd-party libraries.
- [ x] Delete the line that doesn't apply to you in the Acknowledgement section.

## Acknowledgement 
In accordance with the UofA Code of Student Behaviour, we acknowledge that  
(**delete the line that doesn't apply to you**)

- We consulted the following external resources for this assignment.
- https://www.nltk.org/_modules/nltk/tag/brill_trainer.html
- https://www.nltk.org/_modules/nltk/tag/hmm.html
- https://bobbyhadz.com/blog/python-write-list-of-tuples-to-file#:~:text=To%20write%20a%20list%20of,of%20automatically%20closing%20the%20file.
- https://www.geeksforgeeks.org/nlp-brill-tagger/
- We have listed all external resources we consulted for this assignment.

 Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `main.py L:[151:174]` used `argparse` for adding command line arguments to the program.

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`

## Data

The assignment's training data can be found in [data/train.txt](data/train.txt), the in-domain test data can be found in [data/test.txt](data/test.txt), and the out-of-domain test data can be found in [data/test_ood.txt](data/test_ood.txt).
