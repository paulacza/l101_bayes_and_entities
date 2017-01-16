## Enhancing Bernoulli Naive Bayes with Entity-based Knowledge

For the program to run properly the following Python packages need to be installed:
* sklearn
* numpy
* spacy
* nltk

The program is compatible with Python 2.7. The main function accepts 7 parameters:<br />
1. vocabulary size (int) <br />
2. stopwords removal (0/1)<br />
3. use of stemming (0/1)<br />
4. removing HTML tags (0/1)<br />
5. removing headers (0/1)<br />
6. include entity-features (0/1)<br />
7. force using entity-features (0/1)<br />

The main function (tests) can be run with the following command:<br /><br />
python main.py arg1 arg2 arg3 arg4 arg5 arg6 arg7<br /><br />
Such command will make the program read all the WebKB data, divide the data
into 3 folds and run 3 tests each time using different two folds as a training
set. The averaged accuracy and F measure from all of these runs is computed and
printed out to the stdout along with the details about the program parameters.

To run the tests that generated results discussed in the report one should
execute scripts named vocab_size_test.sh and test_entities.sh. Furthermore,
the project contains two additional scripts: param_tuning_tests.sh and
test_entities_force.sh. The first one was used to tune the parameters of
the Naive Bayes classifier, while the latter runs the same tests as the
test_entities.sh script, but forcing the use of entity-related features. 
