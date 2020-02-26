from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from constants import ukbb_gl_constants
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from  PIL import Image
import matplotlib
matplotlib.use('Agg')

G = [0, 255, 0]
R = [255, 0, 0]


# IG Code from  https://github.com/ankurtaly/Integrated-Gradients


def load_image(img_path, sess):
    with open(img_path, 'rb') as f:
        img = f.read()
        img = sess.run(tf.image.decode_png(img))
        return img


def load_model_withbn(m_path):
    graph = tf.Graph()
    cfg = tf.ConfigProto(gpu_options={'allow_growth': True})
    sess = tf.InteractiveSession(graph=graph, config=cfg)
    graph_def = tf.GraphDef.FromString(open(m_path, 'rb').read())
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]
    #
    tf.import_graph_def(graph_def)
    return sess, graph


def integrated_gradients(
        inp,
        target_label_index,
        predictions_and_gradients,
        baseline,
        steps=50,
        gaussian=False):
    if baseline is None:
        if gaussian is False:
            baseline = 0 * inp
        else:
            baseline = np.random.uniform(0.0, 1.0, inp.shape)
    assert (baseline.shape == inp.shape)
    #
    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i) / steps) * (inp - baseline) for i in range(0, steps + 1)]
    predictions, grads = predictions_and_gradients(scaled_inputs,
                                                   target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
    #
    avg_grads = np.average(grads[:-1], axis=0)
    integrated_gradients = (inp - baseline) * avg_grads  # shape: <inp.shape>
    return integrated_gradients, predictions


def random_baseline_integrated_gradients_oct(
        inp,
        target_label_index,
        predictions_and_gradients,
        steps=50,
        num_random_trials=10, ):
    all_intgrads = []
    for i in range(num_random_trials):
        intgrads, prediction_trend = integrated_gradients(
            inp,
            target_label_index=target_label_index,
            predictions_and_gradients=predictions_and_gradients,
            baseline=np.random.random([224, 224, 1]),
            steps=steps, gaussian=True)
        all_intgrads.append(intgrads)
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads


def make_predictions_and_gradients_oct(sess, graph):
    m = 27.5
    s = 32.3
    inp, label_index, predictions, grads = supplement_graph_oct(graph)
    run_graph = sess.make_callable([predictions, grads], feed_list=[inp, label_index])

    #
    def f(images, target_label_index):
        return run_graph([(img - np.array(m)) / np.array(s) for img in images], target_label_index)

    #
    return f


def supplement_graph_oct(graph):
    with graph.as_default():
        label_index = tf.placeholder(tf.int32, [])
        inp = T(graph, 'input_1')
        label_prediction = T(graph, 'dense/BiasAdd')[:, label_index]
        return inp, label_index, T(graph, 'dense/BiasAdd'), tf.gradients(label_prediction, inp)[0]


def T(graph, layer):
    return graph.get_tensor_by_name("import/%s:0" % layer)


def top_label_id_and_score(img, preds_and_grads_fn):
    preds, _ = preds_and_grads_fn([img], 0)
    id = np.argmax(preds[0])
    return id, preds[0]


def Polarity(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise ValueError('Unrecognized polarity option.')


def LinearTransform(attributions,
                    clip_above_percentile=99.9,
                    clip_below_percentile=70.0,
                    low=0.2,
                    plot_distribution=False):
    """Transform the attributions by a linear function.
    Transform the attributions so that the specified percentage of top attribution
    values are mapped to a linear space between `low` and 1.0.
    Args:
      attributions: (numpy.array) The provided attributions.
      percentage: (float) The percentage of top attribution values.
      low: (float) The low end of the linear space.
    Returns:
      (numpy.array) The linearly transformed attributions.
    Raises:
      ValueError: if percentage is not in [0, 100].
    """
    if clip_above_percentile < 0 or clip_above_percentile > 100:
        raise ValueError('clip_above_percentile must be in [0, 100]')
    #
    if clip_below_percentile < 0 or clip_below_percentile > 100:
        raise ValueError('clip_below_percentile must be in [0, 100]')
    #
    if low < 0 or low > 1:
        raise ValueError('low must be in [0, 1]')
    m = ComputeThresholdByTopPercentage(attributions,
                                        percentage=100 - clip_above_percentile,
                                        plot_distribution=plot_distribution)
    e = ComputeThresholdByTopPercentage(attributions,
                                        percentage=100 - clip_below_percentile,
                                        plot_distribution=plot_distribution)
    #
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    # Recover the original sign of the attributions.
    transformed *= np.sign(attributions)
    # Map values below low to 0.
    transformed *= (transformed >= low)
    # Clip values above and below.
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def MorphologicalCleanup(attributions, structure=np.ones((4, 4))):
    closed = ndimage.grey_closing(attributions, structure=structure)
    opened = ndimage.grey_opening(closed, structure=structure)
    return opened


def Outlines(attributions, percentage=99,
             connected_component_structure=np.ones((3, 3)),
             plot_distribution=False):
    # Binarize the attributions mask if not already.
    attributions = Binarize(attributions)
    attributions = ndimage.binary_fill_holes(attributions)
    # Compute connected components of the transformed mask.
    connected_components, num_cc = ndimage.measurements.label(
        attributions, structure=connected_component_structure)
    # Go through each connected component and sum up the attributions of that
    # component.
    overall_sum = np.sum(attributions[connected_components > 0])
    component_sums = []
    for cc_idx in range(1, num_cc + 1):
        cc_mask = connected_components == cc_idx
        component_sum = np.sum(attributions[cc_mask])
        component_sums.append((component_sum, cc_mask))
    # Compute the percentage of top components to keep.
    sorted_sums_and_masks = sorted(
        component_sums, key=lambda x: x[0], reverse=True)
    sorted_sums = list(zip(*sorted_sums_and_masks))[0]
    cumulative_sorted_sums = np.cumsum(sorted_sums)
    cutoff_threshold = percentage * overall_sum / 100
    cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]
    if cutoff_idx > 2:
        cutoff_idx = 2
    #
    # Turn on the kept components.
    border_mask = np.zeros_like(attributions)
    for i in range(cutoff_idx + 1):
        border_mask[sorted_sums_and_masks[i][1]] = 1
    #
    if plot_distribution:
        plt.plot(np.arange(len(sorted_sums)), sorted_sums)
        plt.axvline(x=cutoff_idx)
        plt.show()
    #
    # Hollow out the mask so that only the border is showing.
    eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
    border_mask[eroded_mask] = 0
    return border_mask


def ComputeThresholdByTopPercentage(attributions,
                                    percentage=60.0,
                                    plot_distribution=True):
    """Compute the threshold value that maps to the top percentage of values.
    This function takes the cumulative sum of attributions and computes the set
    of top attributions that contribute to the given percentage of the total sum.
    The lowest value of this given set is returned.
    Args:
      attributions: (numpy.array) The provided attributions.
      percentage: (float) Specified percentage by which to threshold.
      plot_distribution: (bool) If true, plots the distribution of attributions
        and indicates the threshold point by a vertical line.
    Returns:
      (float) The threshold value.
    Raises:
      ValueError: if percentage is not in [0, 100].
    """
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    # For percentage equal to 100, this should in theory return the lowest
    # value as the threshold. However, due to precision errors in numpy's cumsum,
    # the last value won't sum to 100%. Thus, in this special case, we force the
    # threshold to equal the min value.
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    # Sort the attributions from largest to smallest.
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    # Compute a normalized cumulative sum, so that each attribution is mapped to
    # the percentage of the total sum that it and all values above it contribute.
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        # Generate a plot of sorted intgrad scores.
        values_to_plot = np.where(cum_sum >= 95)[0][0]
        values_to_plot = max(values_to_plot, threshold_idx)
        plt.plot(np.arange(values_to_plot), sorted_attributions[:values_to_plot])
        plt.axvline(x=threshold_idx)
        plt.show()
    return threshold


def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def Overlay(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def Binarize(attributions, threshold=0.001):
    return attributions > threshold


def save_as_PIL_img(array, fn, normalize=True):
    if normalize:
        im = Image.fromarray((array * 255.).astype(np.uint8))
    else:
        im = Image.fromarray(array.astype(np.uint8))

    im.save(fn)


def ConvertToGrayscale(attributions):
    return np.average(attributions, axis=2)


def colorize_grads(grads, channel):
    grads = np.expand_dims(grads, 2) * channel
    return grads
