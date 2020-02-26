import logging
import tensorflow as tf
from common.constants import ukbb_gl_constants
from common.common import *
from Eval import run_eval


def run(TRAIN, EVAL, modelname, bs=32, lr=1e-4, ls=True):

    print('Setup training for {}'.format(modelname))
    const = ukbb_gl_constants(modelnm=modelname)
    train_fn = const.tr_gl_nm
    test_fn = const.test_gl_nm
    eval_data_fn, train_inputs, val_inputs, const = setup_training(const, train_fn=train_fn, eval_fn=test_fn, ls=ls,
                                                                   batchsize=bs)
    if TRAIN is True:
        # create model
        logging.info("create model!")
        model = create_model_densnet_no_sobel(lr=lr)
        logging.info("Created model successfully!")

        chpointer = tf.compat.v1.keras.callbacks.ModelCheckpoint(const.checkpoint_path, monitor='val_loss',
                                                                 mode='min',
                                                                 verbose=1, save_best_only=True,
                                                                 save_weights_only=True)

        print('Model_fit!!')
        history = model.fit(train_inputs, epochs=const.epochs, steps_per_epoch=const.num_steps,
                            validation_data=val_inputs, validation_steps=const.val_steps, callbacks=[chpointer])
        print('Model fitted!')
        model.save_weights(const.weights_path)

        save_history(history,const)

    if EVAL is True:
        # eval
        logging.info("Eval model!")

        test_fn=const.te_gl_nm

        run_eval(create_model_densnet_no_sobel(lr=lr), test_fn, const=const, bs=bs, chkpoint=False,fname="test_results.txt")


