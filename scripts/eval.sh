cd ..
MODELNAME=""
DATADIR=""
Project=""


# MLP node classification
for K in 2 4 8 16; do
python embed_eval.py --embed_path ${MODELNAME}/X_embed.torch \
      --data_root_dir ${DATADIR} \
      --project ${Project} --K ${K}  --lr 1e-4 \
       --runs 10
done



# GraphSage node classification
for lr in 0.001; do
for neighbor_sizes in 5_10; do
    for hidden_channels in 256 ; do
      for K in 2 4 8 16; do
  python graphsage_clf.py --embed_path ${MODELNAME}/X_embed.torch \
      --data_root_dir ${DATADIR} \
      --project ${Projects} --K ${K} \
      --epochs 50 --batch_size 1024 --runs 10 --eval_batch_size 1024 \
      --lr ${lr} --hidden_channels ${hidden_channels} --neighbor_sizes ${neighbor_sizes}
done
done
done
done


# Kmeans Clustering
python Kmeans_eval.py   --project ${Project} \
    --embed_path ${LOCALDIR}/${MODELNAME}/X_embed.torch \
    --root_data_dir ${DATADIR}

#Link Prediction
python ogbn_link_eval.py \
    --project ${project} \
    --embed_path ${MODELNAME}/X_embed.torch



