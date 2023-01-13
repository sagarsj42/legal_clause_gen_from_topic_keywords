import os
import time
import json

from haystack.nodes import TfidfRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs


# Install the latest master of Haystack
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

with open('topic-nostep-clause-kwds.c100.k200.lemstop.train.json', 'r') as f:
    train_data = json.load(f)

clause_kwd_limit = 10


def prepare_data(topic_clause_kwds):
    topic_clause_kwds_new = dict()
    for topic, clause_kwds in topic_clause_kwds.items():
        clause_kwds = [(clause, " ".join(kwds[:clause_kwd_limit]))
                       for clause, kwds in clause_kwds]
        topic_clause_kwds_new[topic] = clause_kwds
    return topic_clause_kwds_new


train_data = prepare_data(train_data)

os.makedirs('data')
# ind = 0
for topic, clause_kwds in train_data.items():
    for i, (clause, _) in enumerate(clause_kwds):
        temp = "_".join(topic.split())
        with open("data/"+temp + str(i) + ".txt", "w") as f:
            print(clause, file=f)
        # ind+=1
        # if ind > 100:
        #   break

# Let's first get some files that we want to use
doc_dir = 'data/'

# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir, split_paragraphs=True)
start = time.time()

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)

print('document store ready in : ', time.time() - start)
print(docs[1])


retriever = TfidfRetriever(
    document_store=document_store,
)

with open('topic-nostep-clause-kwds.c100.k200.lemstop.test.json', 'r') as f:
    test_data = json.load(f)

test_data = prepare_data(test_data)
final_results = dict()
final_results['topic'] = list()
final_results['reference'] = list()
final_results['kwds'] = list()
final_results['output'] = list()
for topic, clause_kwds in test_data.items():
    for clause, kwds in clause_kwds:
        result = retriever.retrieve(query=kwds, top_k=1)[0].content
        final_results['topic'].append(topic)
        final_results['kwds'].append(kwds)
        final_results['reference'].append(clause)
        final_results['output'].append(result)

print('Results are ready and are saved')

with open('tfidf_results.json', 'w') as f:
    json.dump(final_results, f)
