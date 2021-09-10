
python -m schema_guided_dst.evaluate \
--dstc8_data_dir dstc8-schema-guided-dialogue \
--prediction_dir large_last/t5-larget5_lr_5e-05_epoch_5_seed_557_neg_num_0.3_canonicalizationTrue0.05/sgd_prediction --eval_set test \
--output_metric_file large_last/t5-larget5_lr_5e-05_epoch_5_seed_557_neg_num_0.3_canonicalizationTrue0.05/sgd_prediction/report.json
