Enhancing Bernoulli Naive Bayes with Entity-based Knowledge

For the program to run properly the following Python packages need to be installed:
sklearn
numpy
spacy
nltk

The program is compatible with Python 2.7. The main function accepts 7 parameters:
1) vocabulary size (int) 
2) stopwords removal (0/1)
3) use of stemming (0/1)
4) removing HTML tags (0/1)
5) removing headers (0/1)
6) include entity-features (0/1)
7) force using entity-features (0/1)

The main function (tests) can be run with the following command:
python main.py arg1 arg2 arg3 arg4 arg5 arg6 arg7
Such command will make the program read all the WebKB data, divide the data
into 3 folds and run 3 tests each time using different two folds as a training
set. The averaged accuracy and F measure from all of these runs is computed and
printed out to the stdout along with the details about the program parameters.

To run the tests that generated results discussed in the report one should
execute scripts named \textit{vocab\_size\_test.sh} and
\textit{test\_entities.sh}. Furthermore, the project contains two additional
scripts: \textit{param\_tuning\_tests.sh} and \textit{test\_entities\_force.sh}.
The first one was used to tune the parameters of the Naive Bayes classifier,
while the latter runs the same tests as the \textit{test\_entities.sh} script,
but forcing the use of entity-related features. 
