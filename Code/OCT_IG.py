from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from common.constants import ukbb_gl_constants
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import common.grad_utils as gu

G=[0, 255, 0.]
R=[255, 0, 0]


def gen_IG(eval_data, pre):
    for i in range(len(eval_data)):
        img_name = eval_data.iloc[i][0]
        full_img_name = img_path + img_name
        aux_gl_c = gu.load_image(full_img_name, sess)
        top_label_id, score = gu.top_label_id_and_score(aux_gl_c, inception_predictions_and_gradients)
        print("Top label: %s, score: %s for image: %s" % (top_label_id, np.array2string(score), img_name))

        attributions = gu.random_baseline_integrated_gradients_oct(aux_gl_c, top_label_id,
                                                                   inception_predictions_and_gradients, steps=50,
                                                                   num_random_trials=10)
        attrs_x = gu.Polarity(attributions, 'positive')
        attrs_x = gu.LinearTransform(attrs_x, 95, 58, 0.0, plot_distribution=False)
        attrs_p = gu.MorphologicalCleanup(np.squeeze(attrs_x))
        attrs_p = gu.Outlines(attrs_p)
        c_attrs_x = np.expand_dims(attrs_p, 2) * G
        combo_attrx = gu.Overlay(c_attrs_x, aux_gl_c)
        fn_sIG = const.save_path + pre+'oct_ig_' + img_name
        gu.save_as_PIL_img(combo_attrx, fn_sIG, normalize=False)



modelname='oct_1'
const = ukbb_gl_constants(modelname)
mpath=const.pb_path+const.model_name +"_final_weights.pb"
sess,graph=gu.load_model_withbn(mpath)
img_path=const.root_dir
inception_predictions_and_gradients = gu.make_predictions_and_gradients_oct(sess, graph)
eval_data=pd.read_table('oct_ig.csv',header=None)
eval_data.columns=['fn']
gen_IG(eval_data, 'all_')