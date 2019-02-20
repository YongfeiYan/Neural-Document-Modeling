#!/bin/bash

#script that computes the observed coherence (pointwise mutual information, normalised pmi or log 
#conditional probability)
#steps:
#1. sample the word counts of the topic words based on the reference corpus
#2. compute the observed coherence using the chosen metric

source activate py2
echo 'python version:'
python -V

if [ $# -ne 3 ]; then
    echo 'topics-file corpus-dir result'
    exit 1
fi

# process topics for the top 10 words
topic_file="`realpath $1`"
ref_corpus_dir="`realpath $2`"

cut -d' ' -f 1-10 $topic_file > $topic_file.10
topic_file=$topic_file.10
wordcount_file="$topic_file.wc.txt"
oc_file="`realpath $3`"

echo 'calculate topic coherence: '$topic_file
cd scripts/topic_interpretability


#parameters
metric="npmi" #evaluation metric: pmi, npmi or lcp
#input
# topic_file="$1"
# ref_corpus_dir="$2"
#output
#wordcount_file="wordcount/wc-oc.txt"
# wordcount_file="$1.wc.txt"
# oc_file="$3"

#compute the word occurrences
echo "Computing word occurrence..."
python ComputeWordCount.py $topic_file $ref_corpus_dir > $wordcount_file

#compute the topic observed coherence
echo "Computing the observed coherence..."
python ComputeObservedCoherence.py $topic_file $metric $wordcount_file -t 5 10 > $oc_file
#python ComputeObservedCoherence.py $topic_file $metric $wordcount_file > $oc_file

echo "rm wordcountt_file"
rm $wordcount_file
tail $oc_file
cd ../..
