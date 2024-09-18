import tensorflow as tf
import keras.backend as K


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """
    a.k.a Smooth Mean Absolute Error
    Huber loss is:
        - less sensitive to outliers in data than the squared error loss
        - differentiable at 0
    """
    error = y_true - y_pred

    squared_loss = 0.5*K.square(error)
    quadratic_loss = 0.5*K.square(clip_delta) + clip_delta*(K.abs(error)-clip_delta)

    return K.mean(
        tf.where(K.abs(error)<=clip_delta, squared_loss, quadratic_loss)
    )

