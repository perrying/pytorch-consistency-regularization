python3 train_val_test.py \
--optimizer adam \
--lr_decay step \
--weight_decay 0 \
--dataset svhn \
--wa f.t.f \
--lr 3e-3 \
--coef 0 \
--out_dir $1 \
--num_labels $2
