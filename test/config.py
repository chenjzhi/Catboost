
train_file= "../dataset/over_script.csv.csv"
test_file= "../dataset/dddd.csv"
catboost_model_path= '../mnt/trained_catboost_model_n1.cbm'
knn_model_path= '../mnt/trained_knn_model.cbm'
text_columns= ['正文', '标题','题材']
force_recompute_embeddings= False
results_dir= '../results'
results_prefix= 'hook_review'
results_suffix= 'catboost_results'
