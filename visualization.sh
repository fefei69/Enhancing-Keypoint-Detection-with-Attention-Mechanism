python oks_plots.py --data output_vgg_f_0603/keypoints.csv output_vgg_f_mha_0605/keypoints.csv output_vgg_f_attlast_0607/keypoints.csv output_vgg_f_transformer_0605/keypoints.csv \
--show --output "vgg_f_mha_0605_PCK" --colours 0 1 2 3 4 --labels "VGG" "VGG-MHA" "VGG-Att" "VGG-Transformer"  

python add_plots.py --data  output_vgg_f_0603/pnp_results.csv output_vgg_f_mha_0605/pnp_results.csv output_vgg_f_attlast_0607/pnp_results.csv output_vgg_f_transformer_0605/pnp_results.csv \
--show --output "vgg_f_mha_0605_ADD" --colours 0 1 2 3 4 --labels "VGG" "VGG-MHA" "VGG-Att" "VGG-Transformer"