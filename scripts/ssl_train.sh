cd ..
LOCALDIR=result/
for mm in  1; do
  for n in 3; do
    for MLM in 0.3 0.6; do
DataDir=""
InitNodeEmbPath=""
bsz=$((64/sample_neighbor_count))
Project=ogbn-products
MODELNAME=${Project}-bert-Neighbor_${neighbor}-sample_${sample_neighbor_count}-LinkPre
python mlm_neighbor.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ${DataDir} \
    --init_node_emb_path ${InitNodeEmbPath} \
    --output_dir  ${LOCALDIR}/${MODELNAME} \
    --per_device_train_batch_size=$bsz \
    --per_device_eval_batch_size=16 \
    --max_seq_length 128 \
    --num_train_epochs 1 \
    --bert_model_name "bert-base-uncased" \
    --is_no_inductive \
    --momentum ${mm} \
    --with_tracking \
    --is_message_agg \
    --neighbor_sizes ${neighbor} \
    --project ${Project} \
    --link_lambda 1 \
    --is_link_pre \
    --sample_neighbor_count ${sample_neighbor_count} \
    --is_lm2many \
    --is_lsp_kd


    # get the embeddings from structure enhanced LM
    python sentence_rep.py --model_name ${LOCALDIR}/${MODELNAME}

done
done
done



