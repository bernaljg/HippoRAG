gpus_available=$1
syn_threshold=0.8

for data in beir_nfcorpus_dev beir_scifact_test;
do
  bash src/setup_hipporag.sh $data facebook/contriever gpt-3.5-turbo-1106 $gpus_available $syn_threshold openai
  bash src/setup_hipporag_colbert.sh $data gpt-3.5-turbo-1106 $gpus_available $syn_threshold openai
done